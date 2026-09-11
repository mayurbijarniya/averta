"""Scores a live session and explains the score.

The model is trained on SWE-Gym OpenHands trajectories and applied here to
Claude Code transcripts, which is a different agent scaffold with a different
tool vocabulary. That transfer is unvalidated — `RiskReport.transfer_warning`
carries the caveat so it cannot be quietly dropped from any output.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from averta.features import FEATURE_NAMES, extract
from averta.features.view import TurnView
from averta.thresholds import GATE_CUT_POINT

DEFAULT_MODEL_PATH = Path("artifacts/model.pkl")

MIN_TURNS_TO_SCORE = 5

TRANSFER_WARNING = (
    "Trained on SWE-Gym OpenHands trajectories; applied to a different agent "
    "scaffold. Cross-scaffold accuracy is unmeasured — treat as indicative."
)

EARLY_WARNING = (
    f"Fewer than {MIN_TURNS_TO_SCORE} turns. Every model scored at chance at "
    "turn 3 in evaluation; no score is reported this early."
)


@dataclass(frozen=True)
class Contribution:
    feature: str
    value: float
    weight: float

    @property
    def effect(self) -> float:
        return self.value * self.weight


NEUTRAL_BAND = 0.03


@dataclass(frozen=True)
class RiskReport:
    session_id: str
    turns: int
    failure_probability: float | None
    drivers: tuple[Contribution, ...]
    features: dict[str, float]
    scored_at_turn: int | None = None
    base_rate: float | None = None
    transfer_warning: str = TRANSFER_WARNING
    note: str | None = None

    @property
    def lift(self) -> float | None:
        """Probability relative to the base rate. 1.0 means no information."""
        if self.failure_probability is None or not self.base_rate:
            return None
        return self.failure_probability / self.base_rate

    @property
    def verdict(self) -> str:
        """Plain reading of the score against the base rate.

        Most sessions in the corpus fail, so a high absolute probability is
        not by itself a warning. What matters is whether this session looks
        worse than the typical one.
        """
        if self.failure_probability is None or self.base_rate is None:
            return "not scored"
        delta = self.failure_probability - self.base_rate
        if abs(delta) <= NEUTRAL_BAND:
            return "no clear signal — indistinguishable from a typical session"
        return "worse than typical" if delta > 0 else "better than typical"

    def render(self) -> str:
        lines = [f"session {self.session_id}", f"turns observed: {self.turns}"]

        if self.failure_probability is None:
            lines.append(f"\n{self.note}")
            return "\n".join(lines)

        if self.scored_at_turn is not None and self.scored_at_turn != self.turns:
            lines.append(f"scored on first {self.scored_at_turn} turns")
        lines.append(f"failure probability: {self.failure_probability:.1%}")
        if self.base_rate is not None:
            lines.append(
                f"corpus base rate:    {self.base_rate:.1%}  "
                f"({self.lift:.2f}x — {self.verdict})"
            )

        if self.drivers:
            lines.append("\nstrongest contributors")
            for item in self.drivers:
                direction = "raises" if item.effect > 0 else "lowers"
                lines.append(
                    f"  {item.feature:<28} {item.value:>10.3f}  {direction} risk"
                )

        lines.append(f"\n{self.transfer_warning}")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)


class Calibrator:
    """Maps raw model scores onto honest probabilities.

    Every estimator here is fitted with class weighting, which is right for
    ranking but leaves the output on a re-balanced scale rather than the true
    one. Measured on the corpus, a raw score of 0.25 corresponded to an
    observed failure rate near 0.70. Rank-based metrics are unaffected, but any
    number shown to a user or returned over MCP has to mean what it says.

    Isotonic regression is fitted on **out-of-fold** scores. Fitting it on
    training scores would learn the model's own overconfidence and report it
    back as calibrated.
    """

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.mapping = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> Calibrator:
        self.mapping.fit(raw_scores, y_true)
        self.fitted = True
        return self

    def apply(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return raw_scores
        return np.clip(self.mapping.predict(raw_scores), 0.0, 1.0)


@dataclass
class Scorer:
    model: object
    feature_names: tuple[str, ...]
    cut_points: tuple[int, ...]
    calibrator: Calibrator | None = None
    base_rate: float | None = None

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> Scorer:
        if not path.exists():
            raise FileNotFoundError(f"no model at {path}; run `averta fit` first")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            model=payload["model"],
            feature_names=tuple(payload["feature_names"]),
            cut_points=tuple(payload["cut_points"]),
            calibrator=payload.get("calibrator"),
            base_rate=payload.get("base_rate"),
        )

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": list(self.feature_names),
                    "cut_points": list(self.cut_points),
                    "calibrator": self.calibrator,
                    "base_rate": self.base_rate,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _weights(self) -> np.ndarray | None:
        """Linear coefficients, scaled back through the standardizer if present."""
        model = self.model
        steps = getattr(model, "named_steps", None)
        if steps is None:
            coefficients = getattr(model, "coef_", None)
            return None if coefficients is None else coefficients[0]

        estimator = steps.get("model")
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return None

        scaler = steps.get("scale")
        scale = getattr(scaler, "scale_", None)
        if scale is None:
            return coefficients[0]
        return coefficients[0] / scale

    def score(self, session_id: str, turns: list[TurnView], top: int = 5) -> RiskReport:
        if len(turns) < MIN_TURNS_TO_SCORE:
            return RiskReport(
                session_id=session_id,
                turns=len(turns),
                failure_probability=None,
                drivers=(),
                features={},
                note=EARLY_WARNING,
            )

        # Features are cumulative over the prefix, so their magnitudes scale
        # with prefix length. Scoring a 900-turn session against a model fitted
        # on prefixes of at most 40 turns puts every count far outside the
        # fitted range and drives the output to a meaningless 1.0. Truncate to
        # the largest evaluated cut the session has reached.
        cut = max((c for c in self.cut_points if c <= len(turns)), default=len(turns))
        scored = turns[:cut]

        features = extract(scored)
        vector = np.array([[features[name] for name in self.feature_names]], dtype=float)
        raw = float(self.model.predict_proba(vector)[0, 1])
        probability = (
            float(self.calibrator.apply(np.array([raw]))[0])
            if self.calibrator is not None
            else raw
        )

        drivers: tuple[Contribution, ...] = ()
        weights = self._weights()
        if weights is not None:
            contributions = [
                Contribution(feature=name, value=features[name], weight=float(weight))
                for name, weight in zip(self.feature_names, weights, strict=True)
            ]
            contributions.sort(key=lambda item: -abs(item.effect))
            drivers = tuple(contributions[:top])

        notes = []
        if cut < GATE_CUT_POINT:
            notes.append(
                f"Scored at turn {cut}, before the evaluated cut point of "
                f"{GATE_CUT_POINT}; accuracy is lower this early."
            )
        if len(turns) > cut:
            notes.append(
                f"Session has {len(turns)} turns but the model is fitted only up to "
                f"{max(self.cut_points)}. This score reflects the first {cut} turns."
            )

        return RiskReport(
            session_id=session_id,
            turns=len(turns),
            scored_at_turn=cut,
            base_rate=self.base_rate,
            failure_probability=probability,
            drivers=drivers,
            features=features,
            note=" ".join(notes) or None,
        )


def fit_and_save(
    X: np.ndarray,
    y: np.ndarray,
    cut_points: tuple[int, ...],
    model_name: str = "logistic",
    path: Path = DEFAULT_MODEL_PATH,
    calibration_scores: np.ndarray | None = None,
    calibration_labels: np.ndarray | None = None,
) -> Scorer:
    """Fit the final model and, when given out-of-fold scores, calibrate it.

    `calibration_scores` must be out-of-fold. Passing the model's own training
    predictions would fit the calibrator to its overconfidence.
    """
    from averta.models import build_registry

    model = build_registry(FEATURE_NAMES)[model_name](y)
    model.fit(X, y)

    calibrator = None
    if calibration_scores is not None and calibration_labels is not None:
        calibrator = Calibrator().fit(calibration_scores, calibration_labels)

    scorer = Scorer(
        model=model,
        feature_names=FEATURE_NAMES,
        cut_points=cut_points,
        calibrator=calibrator,
        base_rate=float(np.mean(y)),
    )
    scorer.save(path)
    return scorer
