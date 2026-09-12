"""Evaluation metrics, implemented directly rather than imported.

Accuracy is deliberately absent. The positive class is rare, so a model that
predicts the majority class everywhere scores well on accuracy while being
useless. Every metric here is either rank-based or reported at a fixed
operating point.

Confidence intervals resample **groups**, not rows. Sessions from the same
repository are correlated, and resampling rows would understate the interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _as_arrays(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    if truth.shape != score.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {score.shape}")
    if truth.size == 0:
        raise ValueError("empty input")
    return truth, score


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under the ROC curve via the rank-sum identity.

    Equivalent to the probability that a randomly chosen positive outranks a
    randomly chosen negative. Ties receive averaged ranks.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = truth == 1
    n_pos = int(positives.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=float)
    ranks[order] = np.arange(1, score.size + 1, dtype=float)

    # Average ranks within tied score groups.
    sorted_scores = score[order]
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                tied = order[start:index]
                ranks[tied] = ranks[tied].mean()
            start = index

    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Average precision, the step-wise area under the precision/recall curve."""
    truth, score = _as_arrays(y_true, y_score)
    n_pos = int((truth == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    truth = truth[order]

    true_positives = np.cumsum(truth)
    predicted = np.arange(1, truth.size + 1, dtype=float)
    precision = true_positives / predicted
    recall = true_positives / n_pos

    recall_gain = np.diff(recall, prepend=0.0)
    return float((precision * recall_gain).sum())


def roc_curve(
    y_true: Sequence[int], y_score: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fpr, tpr, threshold) with a leading (0, 0) point."""
    truth, score = _as_arrays(y_true, y_score)
    n_pos = (truth == 1).sum()
    n_neg = truth.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs both classes present")

    order = np.argsort(-score, kind="mergesort")
    truth, score = truth[order], score[order]

    tps = np.cumsum(truth)
    fps = np.cumsum(1 - truth)

    # Keep only the last index of each tied score run.
    distinct = np.r_[np.nonzero(np.diff(score))[0], score.size - 1]

    tpr = np.r_[0.0, tps[distinct] / n_pos]
    fpr = np.r_[0.0, fps[distinct] / n_neg]
    thresholds = np.r_[np.inf, score[distinct]]
    return fpr, tpr, thresholds


def recall_at_fpr(
    y_true: Sequence[int], y_score: Sequence[float], target_fpr: float = 0.05
) -> tuple[float, float]:
    """Best achievable recall without exceeding `target_fpr`.

    Returns (recall, threshold). This is the operating point that matters: the
    monitor may only intervene if it rarely stops sessions that would succeed.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    allowed = fpr <= target_fpr
    if not allowed.any():
        return 0.0, float("inf")
    index = int(np.flatnonzero(allowed)[-1])
    return float(tpr[index]), float(thresholds[index])


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    truth, prob = _as_arrays(y_true, y_prob)
    return float(np.mean((prob - truth) ** 2))


def calibration_curve(
    y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean_predicted, observed_rate, count) per occupied bin."""
    truth, prob = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)

    predicted, observed, counts = [], [], []
    for index in range(bins):
        mask = assignment == index
        if not mask.any():
            continue
        predicted.append(prob[mask].mean())
        observed.append(truth[mask].mean())
        counts.append(mask.sum())

    return np.array(predicted), np.array(observed), np.array(counts)


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def grouped_bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    groups: Sequence[str],
    metric=auroc,
    samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Interval:
    """Percentile bootstrap interval, resampling whole groups with replacement.

    Rows within a repository are not independent, so the unit of resampling is
    the repository. Replicates where a class is absent are discarded.
    """
    truth, score = _as_arrays(y_true, y_score)
    group_array = np.asarray(groups)

    unique = np.unique(group_array)
    indices_by_group = {name: np.flatnonzero(group_array == name) for name in unique}

    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(samples):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        picked = np.concatenate([indices_by_group[name] for name in drawn])
        replicate_truth = truth[picked]
        if replicate_truth.min() == replicate_truth.max():
            continue
        value = metric(replicate_truth, score[picked])
        if not np.isnan(value):
            estimates.append(value)

    point = metric(truth, score)
    if not estimates:
        return Interval(point=point, lower=float("nan"), upper=float("nan"))

    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return Interval(point=float(point), lower=float(lower), upper=float(upper))
