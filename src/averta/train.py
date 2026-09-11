"""Cross-validated evaluation at a single cut point.

Predictions are pooled out-of-fold: every row is scored exactly once, by a
model that never saw its repository during training. Metrics are computed on
that pooled vector, and confidence intervals resample repositories.

**Label polarity.** The positive class is *failure* — the session does not
resolve. That is the event the tool warns about, and it makes the false
positive rate mean "sessions that would have succeeded but were flagged",
which is the safety quantity the gate constrains.

One metric deliberately uses the other polarity. Average precision is
informative about the *rare* class, and here failure is the majority at
roughly nine in ten. `auprc_minority` is therefore computed on resolution as
the positive class, matching the base rate the threshold was pre-registered
against. Both are reported with their polarity named.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

from averta.features import FEATURE_NAMES
from averta.metrics import auprc, auroc, brier_score, grouped_bootstrap_ci, recall_at_fpr
from averta.models import build_registry
from averta.splits import check_folds, describe, grouped_k_fold
from averta.thresholds import (
    BOOTSTRAP_SAMPLES,
    TARGET_FPR,
    GateResult,
)


@dataclass
class Dataset:
    X: np.ndarray
    y_fail: np.ndarray
    groups: np.ndarray
    session_ids: np.ndarray
    cut_point: int

    @property
    def y_resolve(self) -> np.ndarray:
        return 1 - self.y_fail

    @property
    def failure_rate(self) -> float:
        return float(self.y_fail.mean())

    @property
    def resolve_rate(self) -> float:
        return float(self.y_resolve.mean())

    def __len__(self) -> int:
        return len(self.y_fail)


@dataclass
class ModelResult:
    name: str
    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    auprc_minority: float
    auprc_lift: float
    recall_at_fpr: float
    threshold: float
    brier: float
    fit_seconds: float
    predictions: np.ndarray = field(repr=False)


def load(db: str, cut_point: int) -> Dataset:
    conn = duckdb.connect(db, read_only=True)
    try:
        columns = ", ".join(FEATURE_NAMES)
        rows = conn.execute(
            f"SELECT session_id, repo, outcome, {columns} "
            "FROM prefix_features WHERE cut_point = ? ORDER BY session_id",
            [cut_point],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"no rows at cut point {cut_point}; run `averta features` first")

    session_ids = np.array([row[0] for row in rows])
    groups = np.array([row[1] for row in rows])
    # outcome is `resolved`; the modelling target is its complement.
    y_fail = np.array([0 if bool(row[2]) else 1 for row in rows])
    X = np.array([row[3:] for row in rows], dtype=float)

    return Dataset(
        X=X, y_fail=y_fail, groups=groups, session_ids=session_ids, cut_point=cut_point
    )


def cross_validate(data: Dataset, n_splits: int = 5) -> tuple[list[ModelResult], dict[str, Any]]:
    # Folds are balanced on the rare class, which is resolution, not failure.
    folds = grouped_k_fold(data.groups, data.y_resolve, n_splits)
    warnings = check_folds(folds, data.y_resolve)
    registry = build_registry(FEATURE_NAMES)

    results = []
    for name, factory in registry.items():
        p_fail = np.zeros(len(data), dtype=float)
        elapsed = 0.0

        for fold in folds:
            model = factory(data.y_fail[fold.train])
            start = time.perf_counter()
            model.fit(data.X[fold.train], data.y_fail[fold.train])
            elapsed += time.perf_counter() - start
            p_fail[fold.test] = model.predict_proba(data.X[fold.test])[:, 1]

        interval = grouped_bootstrap_ci(
            data.y_fail, p_fail, data.groups, metric=auroc, samples=BOOTSTRAP_SAMPLES
        )
        # Average precision on the minority class: resolution, scored by 1 - p(fail).
        precision = auprc(data.y_resolve, 1.0 - p_fail)
        recall, threshold = recall_at_fpr(data.y_fail, p_fail, TARGET_FPR)

        results.append(
            ModelResult(
                name=name,
                auroc=interval.point,
                auroc_ci_lower=interval.lower,
                auroc_ci_upper=interval.upper,
                auprc_minority=precision,
                auprc_lift=precision / data.resolve_rate if data.resolve_rate else float("nan"),
                recall_at_fpr=recall,
                threshold=threshold,
                brier=brier_score(data.y_fail, p_fail),
                fit_seconds=elapsed,
                predictions=p_fail,
            )
        )

    diagnostics = {
        "rows": len(data),
        "failure_rate": round(data.failure_rate, 4),
        "resolve_rate": round(data.resolve_rate, 4),
        "positive_class": "failure (session does not resolve)",
        "auprc_polarity": "minority class = resolution",
        "n_splits": n_splits,
        "folds": describe(folds, data.y_resolve),
        "warnings": warnings,
    }
    return results, diagnostics


def gate(results: list[ModelResult]) -> tuple[ModelResult, GateResult]:
    """Applies the pre-registered criteria to the best non-baseline model."""
    baselines = {"majority", "turn_index_only", "error_repeat_only"}
    candidates = [r for r in results if r.name not in baselines]
    best = max(candidates, key=lambda r: r.auroc)

    turn_baseline = next(r for r in results if r.name == "turn_index_only")
    beats_baseline = best.auroc_ci_lower > turn_baseline.auroc_ci_upper

    return best, GateResult(
        auroc=best.auroc,
        auroc_ci_lower=best.auroc_ci_lower,
        auprc=best.auprc_minority,
        recall_at_target_fpr=best.recall_at_fpr,
        beats_turn_baseline=beats_baseline,
    )


def render_table(results: list[ModelResult], resolve_rate: float) -> str:
    header = (
        f"{'model':<24}{'AUROC':>7}{'95% CI':>18}{'AUPRC':>8}{'lift':>7}"
        f"{'R@5%FPR':>9}{'Brier':>8}{'fit s':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.name:<24}"
            f"{result.auroc:>7.3f}"
            f"{f'[{result.auroc_ci_lower:.3f}, {result.auroc_ci_upper:.3f}]':>18}"
            f"{result.auprc_minority:>8.3f}"
            f"{result.auprc_lift:>7.2f}"
            f"{result.recall_at_fpr:>9.3f}"
            f"{result.brier:>8.4f}"
            f"{result.fit_seconds:>8.2f}"
        )
    lines.append("")
    lines.append("positive class = failure. AUROC and R@5%FPR use that polarity:")
    lines.append("  R@5%FPR = share of doomed sessions caught while wrongly")
    lines.append("  flagging at most 5% of sessions that would have resolved.")
    lines.append(
        f"AUPRC is on the minority class (resolution, base rate {resolve_rate:.2%}); "
        "lift is relative to it."
    )
    return "\n".join(lines)
