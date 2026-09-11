"""Cross-scaffold feature comparison.

Phase 6 was planned as a transfer evaluation: train on SWE-Gym OpenHands
trajectories, test on hand-labelled Claude Code sessions, report the AUROC gap.
That is not possible. Only a handful of local transcripts exist and none carry
an outcome label, so any transfer AUROC would be computed on a sample far too
small to mean anything.

What *is* measurable without labels is whether the features compute
comparably at all. If `tool_entropy` on a Claude Code session lands in a range
the model never saw during training, its output is extrapolation regardless of
how good the model is. That is a real and reportable finding, and it is the
prerequisite for transfer rather than a substitute for measuring it.

Every result here is reported with its sample size attached, because the
sample is small enough that the sample size is the headline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SAMPLES_FOR_INFERENCE = 30


@dataclass(frozen=True)
class FeatureShift:
    feature: str
    corpus_mean: float
    corpus_sd: float
    local_mean: float
    local_sd: float
    standardized_difference: float
    inside_corpus_range: bool

    @property
    def magnitude(self) -> str:
        """Conventional reading of a standardized mean difference."""
        size = abs(self.standardized_difference)
        if size < 0.2:
            return "negligible"
        if size < 0.5:
            return "small"
        if size < 0.8:
            return "moderate"
        return "large"


@dataclass(frozen=True)
class DriftReport:
    n_corpus: int
    n_local: int
    shifts: tuple[FeatureShift, ...]

    @property
    def underpowered(self) -> bool:
        return self.n_local < MIN_SAMPLES_FOR_INFERENCE

    @property
    def out_of_range(self) -> tuple[FeatureShift, ...]:
        return tuple(s for s in self.shifts if not s.inside_corpus_range)

    def render(self, top: int = 12) -> str:
        lines = [
            f"corpus sessions: {self.n_corpus}",
            f"local sessions:  {self.n_local}",
            "",
        ]

        if self.underpowered:
            lines += [
                f"UNDERPOWERED — {self.n_local} local sessions is below the "
                f"{MIN_SAMPLES_FOR_INFERENCE} needed for any inference.",
                "These numbers describe this handful of sessions and nothing more.",
                "They are not an estimate of cross-scaffold behaviour.",
                "",
            ]

        header = f"{'feature':<30}{'corpus':>10}{'local':>10}{'std diff':>10}  magnitude"
        lines += [header, "-" * (len(header) + 8)]

        ordered = sorted(self.shifts, key=lambda s: -abs(s.standardized_difference))
        for shift in ordered[:top]:
            flag = "  OUT OF RANGE" if not shift.inside_corpus_range else ""
            lines.append(
                f"{shift.feature:<30}"
                f"{shift.corpus_mean:>10.2f}"
                f"{shift.local_mean:>10.2f}"
                f"{shift.standardized_difference:>10.2f}"
                f"  {shift.magnitude}{flag}"
            )

        if self.out_of_range:
            lines += [
                "",
                f"{len(self.out_of_range)} feature(s) fall outside the range seen "
                "in training. For those, the model extrapolates.",
            ]

        return "\n".join(lines)


def compare(
    feature_names: tuple[str, ...],
    corpus: np.ndarray,
    local: np.ndarray,
) -> DriftReport:
    """Standardized mean difference per feature, corpus against local.

    Positive means the local sessions score higher. The corpus standard
    deviation is the denominator — the local sample is too small to estimate
    a pooled one.
    """
    if corpus.shape[1] != len(feature_names) or local.shape[1] != len(feature_names):
        raise ValueError("feature count mismatch")

    shifts = []
    for index, name in enumerate(feature_names):
        corpus_column = corpus[:, index]
        local_column = local[:, index]

        corpus_mean = float(corpus_column.mean())
        corpus_sd = float(corpus_column.std())
        local_mean = float(local_column.mean())
        local_sd = float(local_column.std()) if local_column.size > 1 else 0.0

        difference = (
            (local_mean - corpus_mean) / corpus_sd if corpus_sd > 0 else 0.0
        )

        low, high = float(corpus_column.min()), float(corpus_column.max())
        inside = bool(local_column.min() >= low and local_column.max() <= high)

        shifts.append(
            FeatureShift(
                feature=name,
                corpus_mean=corpus_mean,
                corpus_sd=corpus_sd,
                local_mean=local_mean,
                local_sd=local_sd,
                standardized_difference=difference,
                inside_corpus_range=inside,
            )
        )

    return DriftReport(
        n_corpus=int(corpus.shape[0]),
        n_local=int(local.shape[0]),
        shifts=tuple(shifts),
    )
