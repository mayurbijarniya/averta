"""Grouped cross-validation.

Splits are grouped by repository: every row from a repository lands in exactly
one fold. Rows sharing a repository are not independent — the same codebase,
the same test suite, often the same underlying issue attempted repeatedly — so
splitting by row would let a model memorize a repository in training and be
rewarded for it at test time.

The corpus has few repositories (11 as measured in phase 0) and a rare positive
class, so folds are assembled greedily by positive count rather than by row
count. Balancing on rows alone can leave a fold with too few positives for
AUPRC or recall at a fixed FPR to mean anything.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[str, ...]

    def __len__(self) -> int:
        return self.test.size


class InsufficientGroups(ValueError):
    pass


def group_positive_counts(
    groups: Sequence[str], y: Sequence[int]
) -> dict[str, tuple[int, int]]:
    """Maps each group to (row count, positive count)."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for group, label in zip(groups, y, strict=True):
        tally[group][0] += 1
        tally[group][1] += int(bool(label))
    return {group: (rows, positives) for group, (rows, positives) in tally.items()}


def assign_groups_to_folds(
    groups: Sequence[str], y: Sequence[int], n_splits: int
) -> list[list[str]]:
    """Greedy balancing: the largest positive block goes to the emptiest fold."""
    counts = group_positive_counts(groups, y)
    if len(counts) < n_splits:
        raise InsufficientGroups(
            f"{len(counts)} groups cannot fill {n_splits} folds"
        )

    ordered = sorted(counts.items(), key=lambda item: (-item[1][1], -item[1][0], item[0]))

    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    loads = [[0, 0] for _ in range(n_splits)]  # [positives, rows]

    for group, (rows, positives) in ordered:
        target = min(range(n_splits), key=lambda i: (loads[i][0], loads[i][1]))
        buckets[target].append(group)
        loads[target][0] += positives
        loads[target][1] += rows

    return buckets


def grouped_k_fold(
    groups: Sequence[str], y: Sequence[int], n_splits: int = 5
) -> list[Fold]:
    group_array = np.asarray(groups)
    buckets = assign_groups_to_folds(groups, y, n_splits)

    folds = []
    for index, held_out in enumerate(buckets):
        mask = np.isin(group_array, held_out)
        folds.append(
            Fold(
                index=index,
                train=np.flatnonzero(~mask),
                test=np.flatnonzero(mask),
                test_groups=tuple(sorted(held_out)),
            )
        )
    return folds


def describe(folds: Sequence[Fold], y: Sequence[int]) -> list[dict[str, object]]:
    labels = np.asarray(y)
    rows = []
    for fold in folds:
        test_labels = labels[fold.test]
        positives = int(test_labels.sum())
        rows.append(
            {
                "fold": fold.index,
                "train_rows": int(fold.train.size),
                "test_rows": int(fold.test.size),
                "test_positives": positives,
                "test_positive_rate": round(float(test_labels.mean()), 4)
                if test_labels.size
                else None,
                "repos": len(fold.test_groups),
            }
        )
    return rows


def check_folds(folds: Sequence[Fold], y: Sequence[int], min_positives: int = 10) -> list[str]:
    """Returns human-readable warnings about folds too small to evaluate on."""
    labels = np.asarray(y)
    warnings = []
    for fold in folds:
        test_labels = labels[fold.test]
        positives = int(test_labels.sum())
        if test_labels.size == 0:
            warnings.append(f"fold {fold.index} has no test rows")
        elif positives == 0:
            warnings.append(f"fold {fold.index} has no positives; ranking metrics undefined")
        elif positives < min_positives:
            warnings.append(
                f"fold {fold.index} has only {positives} positives; estimates will be noisy"
            )
    return warnings
