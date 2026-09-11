import numpy as np
import pytest

from averta.splits import (
    InsufficientGroups,
    assign_groups_to_folds,
    check_folds,
    describe,
    group_positive_counts,
    grouped_k_fold,
)


def corpus(repos: int = 11, per_repo: int = 40, positive_rate: float = 0.08, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups, labels = [], []
    for repo in range(repos):
        groups.extend([f"repo{repo}"] * per_repo)
        labels.extend((rng.random(per_repo) < positive_rate).astype(int))
    return groups, np.array(labels)


class TestGroupPositiveCounts:
    def test_counts_rows_and_positives(self):
        counts = group_positive_counts(["a", "a", "b"], [1, 0, 1])
        assert counts == {"a": (2, 1), "b": (1, 1)}

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            group_positive_counts(["a", "b"], [1])


class TestAssignment:
    def test_every_group_assigned_exactly_once(self):
        groups, labels = corpus()
        buckets = assign_groups_to_folds(groups, labels, 5)
        flattened = [group for bucket in buckets for group in bucket]
        assert sorted(flattened) == sorted(set(groups))

    def test_too_few_groups_is_rejected(self):
        with pytest.raises(InsufficientGroups):
            assign_groups_to_folds(["a", "b"], [0, 1], 5)

    def test_positives_are_spread_across_folds(self):
        # One repo holds every positive; it cannot be split, but the remaining
        # folds must still receive the other repos.
        groups = [f"repo{i}" for i in range(6) for _ in range(10)]
        labels = [1 if group == "repo0" else 0 for group in groups]
        buckets = assign_groups_to_folds(groups, labels, 3)
        assert all(bucket for bucket in buckets)

    def test_balances_positive_counts(self):
        groups = [f"repo{i}" for i in range(6) for _ in range(10)]
        labels = [1 if i % 10 < 3 else 0 for i, _ in enumerate(groups)]
        buckets = assign_groups_to_folds(groups, labels, 3)
        counts = group_positive_counts(groups, labels)
        loads = [sum(counts[g][1] for g in bucket) for bucket in buckets]
        assert max(loads) - min(loads) <= max(counts[g][1] for g in counts)


class TestGroupedKFold:
    def test_train_and_test_are_disjoint(self):
        groups, labels = corpus()
        for fold in grouped_k_fold(groups, labels, 5):
            assert set(fold.train).isdisjoint(set(fold.test))

    def test_every_row_is_tested_exactly_once(self):
        groups, labels = corpus()
        folds = grouped_k_fold(groups, labels, 5)
        tested = np.concatenate([fold.test for fold in folds])
        assert sorted(tested) == list(range(len(groups)))

    def test_no_repo_appears_in_both_sides(self):
        groups, labels = corpus()
        group_array = np.asarray(groups)
        for fold in grouped_k_fold(groups, labels, 5):
            train_repos = set(group_array[fold.train])
            test_repos = set(group_array[fold.test])
            assert train_repos.isdisjoint(test_repos)

    def test_train_plus_test_covers_everything(self):
        groups, labels = corpus()
        for fold in grouped_k_fold(groups, labels, 5):
            assert fold.train.size + fold.test.size == len(groups)

    def test_deterministic(self):
        groups, labels = corpus()
        first = [f.test_groups for f in grouped_k_fold(groups, labels, 5)]
        second = [f.test_groups for f in grouped_k_fold(groups, labels, 5)]
        assert first == second


class TestDiagnostics:
    def test_describe_reports_every_fold(self):
        groups, labels = corpus()
        rows = describe(grouped_k_fold(groups, labels, 5), labels)
        assert len(rows) == 5
        assert sum(row["test_rows"] for row in rows) == len(groups)

    def test_warns_when_a_fold_has_no_positives(self):
        groups = [f"repo{i}" for i in range(4) for _ in range(10)]
        labels = [1 if group == "repo0" else 0 for group in groups]
        warnings = check_folds(grouped_k_fold(groups, labels, 4), labels)
        assert any("no positives" in warning for warning in warnings)

    def test_warns_when_positives_are_scarce(self):
        groups, labels = corpus(repos=11, per_repo=20, positive_rate=0.05, seed=1)
        warnings = check_folds(grouped_k_fold(groups, labels, 5), labels, min_positives=10)
        assert warnings

    def test_silent_when_folds_are_healthy(self):
        groups, labels = corpus(repos=11, per_repo=400, positive_rate=0.2, seed=2)
        assert check_folds(grouped_k_fold(groups, labels, 5), labels) == []
