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


@dataclass(frozen=True)
class RiskReport:
    session_id: str
    turns: int
    failure_probability: float | None
    drivers: tuple[Contribution, ...]
    features: dict[str, float]
    scored_at_turn: int | None = None
    transfer_warning: str = TRANSFER_WARNING
    note: str | None = None

    def render(self) -> str:
        lines = [f"session {self.session_id}", f"turns observed: {self.turns}"]

        if self.failure_probability is None:
            lines.append(f"\n{self.note}")
            return "\n".join(lines)

        if self.scored_at_turn is not None and self.scored_at_turn != self.turns:
            lines.append(f"scored on first {self.scored_at_turn} turns")
        lines.append(f"failure probability: {self.failure_probability:.1%}")

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


@dataclass
class Scorer:
    model: object
    feature_names: tuple[str, ...]
    cut_points: tuple[int, ...]

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
        )

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": list(self.feature_names),
                    "cut_points": list(self.cut_points),
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
        probability = float(self.model.predict_proba(vector)[0, 1])

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
) -> Scorer:
    from averta.models import build_registry

    model = build_registry(FEATURE_NAMES)[model_name](y)
    model.fit(X, y)
    scorer = Scorer(model=model, feature_names=FEATURE_NAMES, cut_points=cut_points)
    scorer.save(path)
    return scorer
