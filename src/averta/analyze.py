"""Phase 4 diagnostics: importance, calibration, and inference cost.

Permutation importance is computed under the same repo-grouped folds used for
scoring. Importance measured on training data, or on a random split, would
reward features that identify a repository rather than a failing trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from averta.features import FEATURE_NAMES
from averta.metrics import auroc, calibration_curve
from averta.models import build_registry
from averta.splits import grouped_k_fold
from averta.train import Dataset

REPEATS = 8
SEED = 17


@dataclass(frozen=True)
class Importance:
    feature: str
    drop: float
    drop_std: float


def permutation_importance(
    data: Dataset, model_name: str, n_splits: int = 5, repeats: int = REPEATS
) -> list[Importance]:
    """AUROC lost when one feature is shuffled within the held-out fold."""
    folds = grouped_k_fold(data.groups, data.y_resolve, n_splits)
    factory = build_registry(FEATURE_NAMES)[model_name]
    rng = np.random.default_rng(SEED)

    fitted = []
    for fold in folds:
        model = factory(data.y_fail[fold.train])
        model.fit(data.X[fold.train], data.y_fail[fold.train])
        fitted.append((fold, model))

    def pooled_auroc(permute: int | None) -> float:
        scores = np.zeros(len(data), dtype=float)
        for fold, model in fitted:
            block = data.X[fold.test].copy()
            if permute is not None:
                block[:, permute] = rng.permutation(block[:, permute])
            scores[fold.test] = model.predict_proba(block)[:, 1]
        return auroc(data.y_fail, scores)

    reference = pooled_auroc(None)

    results = []
    for index, name in enumerate(FEATURE_NAMES):
        drops = [reference - pooled_auroc(index) for _ in range(repeats)]
        results.append(
            Importance(
                feature=name,
                drop=float(np.mean(drops)),
                drop_std=float(np.std(drops)),
            )
        )

    return sorted(results, key=lambda item: -item.drop)


def calibration(data: Dataset, predictions: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    predicted, observed, counts = calibration_curve(data.y_fail, predictions, bins=bins)
    return {
        "predicted": predicted.tolist(),
        "observed": observed.tolist(),
        "counts": counts.astype(int).tolist(),
    }


@dataclass(frozen=True)
class CostProfile:
    model: str
    fit_seconds: float
    predict_single_ms_p50: float
    predict_single_ms_p95: float
    predict_batch_ms_per_1k: float
    parameters_bytes: int


def _pickled_size(model) -> int:
    import pickle

    return len(pickle.dumps(model))


def inference_cost(
    data: Dataset, model_name: str, trials: int = 200
) -> CostProfile:
    """Single-row latency is the deployment-relevant number.

    The monitor scores one live session per turn, so batch throughput is not
    what matters; the cost of one prediction is.
    """
    factory = build_registry(FEATURE_NAMES)[model_name]
    model = factory(data.y_fail)

    start = time.perf_counter()
    model.fit(data.X, data.y_fail)
    fit_seconds = time.perf_counter() - start

    single = data.X[:1]
    model.predict_proba(single)  # warm any lazy initialisation

    timings = []
    for _ in range(trials):
        begin = time.perf_counter()
        model.predict_proba(single)
        timings.append((time.perf_counter() - begin) * 1000)

    batch = data.X[: min(1000, len(data))]
    begin = time.perf_counter()
    model.predict_proba(batch)
    batch_ms = (time.perf_counter() - begin) * 1000 * (1000 / len(batch))

    return CostProfile(
        model=model_name,
        fit_seconds=round(fit_seconds, 4),
        predict_single_ms_p50=round(float(np.percentile(timings, 50)), 4),
        predict_single_ms_p95=round(float(np.percentile(timings, 95)), 4),
        predict_batch_ms_per_1k=round(batch_ms, 3),
        parameters_bytes=_pickled_size(model),
    )


def render_importance(items: list[Importance], top: int = 15) -> str:
    lines = [f"{'feature':<32}{'AUROC drop':>12}{'sd':>8}", "-" * 52]
    for item in items[:top]:
        lines.append(f"{item.feature:<32}{item.drop:>12.4f}{item.drop_std:>8.4f}")
    return "\n".join(lines)


def out_of_fold_predictions(
    data: Dataset, model_name: str, n_splits: int = 5
) -> np.ndarray:
    """Pooled out-of-fold failure probabilities, for downstream simulation."""
    folds = grouped_k_fold(data.groups, data.y_resolve, n_splits)
    factory = build_registry(FEATURE_NAMES)[model_name]

    scores = np.zeros(len(data.y_fail), dtype=float)
    for fold in folds:
        model = factory(data.y_fail[fold.train])
        model.fit(data.X[fold.train], data.y_fail[fold.train])
        scores[fold.test] = model.predict_proba(data.X[fold.test])[:, 1]
    return scores


def render_cost(profiles: list[CostProfile]) -> str:
    header = (
        f"{'model':<24}{'fit s':>8}{'p50 ms':>9}{'p95 ms':>9}"
        f"{'ms/1k':>9}{'size KB':>10}"
    )
    lines = [header, "-" * len(header)]
    for profile in profiles:
        lines.append(
            f"{profile.model:<24}"
            f"{profile.fit_seconds:>8.3f}"
            f"{profile.predict_single_ms_p50:>9.3f}"
            f"{profile.predict_single_ms_p95:>9.3f}"
            f"{profile.predict_batch_ms_per_1k:>9.2f}"
            f"{profile.parameters_bytes / 1024:>10.1f}"
        )
    return "\n".join(lines)
