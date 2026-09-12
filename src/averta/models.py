"""Model definitions, including the trivial baselines.

The baselines are not decoration. With a positive class near one in ten, a
model that looks strong on any threshold-free metric must still be shown to
beat predicting the base rate everywhere, and to beat using the turn index
alone. Phase 0 measured turn count as nearly outcome-independent, so the
turn-index baseline is expected to sit close to chance, it is reported anyway
so that expectation is evidenced rather than asserted.

Every estimator carries class weighting. Unweighted fits on this corpus
collapse toward always predicting failure.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_SEED = 17


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class MajorityBaseline:
    """Emits one global constant for every row, so AUROC is exactly 0.5.

    Deliberately *not* each fold's training base rate. Out-of-fold predictions
    are pooled before scoring, and per-fold constants differ slightly, which
    injects fold-level base-rate variation into the pooled vector as ranking
    signal. That produced an AUROC of 0.382 on a predictor that by
    construction knows nothing.
    """

    CONSTANT = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> MajorityBaseline:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        column = np.full(len(X), self.CONSTANT)
        return np.column_stack([1 - column, column])


class SingleFeatureBaseline:
    """Logistic regression on one feature, selected by name."""

    def __init__(self, feature_index: int) -> None:
        self.feature_index = feature_index
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SingleFeatureBaseline:
        self.model.fit(X[:, [self.feature_index]], y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, [self.feature_index]])


def logistic() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def hist_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def positive_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, for `scale_pos_weight`."""
    positives = float(np.sum(y))
    if positives == 0:
        return 1.0
    return float(len(y) - positives) / positives


def build_registry(feature_names: tuple[str, ...]) -> dict[str, callable]:
    """Model factories keyed by name. Baselines first, so they print first."""
    turn_index = feature_names.index("turns_seen")
    error_index = feature_names.index("max_error_repeat")

    return {
        "majority": lambda y: MajorityBaseline(),
        "turn_index_only": lambda y: SingleFeatureBaseline(turn_index),
        "error_repeat_only": lambda y: SingleFeatureBaseline(error_index),
        "logistic": lambda y: logistic(),
        "random_forest": lambda y: random_forest(),
        "hist_gradient_boosting": lambda y: hist_gradient_boosting(),
        "xgboost": lambda y: xgboost(positive_weight(y)),
    }
