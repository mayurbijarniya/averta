import math

import numpy as np
import pytest

from averta.metrics import (
    auprc,
    auroc,
    brier_score,
    calibration_curve,
    grouped_bootstrap_ci,
    recall_at_fpr,
    roc_curve,
)


class TestAuroc:
    def test_perfect_separation(self):
        assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_perfectly_inverted(self):
        assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

    def test_all_scores_tied_is_chance(self):
        assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5

    def test_known_value(self):
        # Positives score 0.4 and 0.8; negatives 0.1 and 0.5. The negative at
        # 0.5 outranks the positive at 0.4, losing one of four pairs.
        assert auroc([0, 1, 0, 1], [0.1, 0.4, 0.5, 0.8]) == 0.75

    def test_all_pairs_won_is_one(self):
        assert auroc([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8]) == 1.0

    def test_matches_rank_definition_on_random_data(self):
        rng = np.random.default_rng(3)
        truth = rng.integers(0, 2, size=200)
        score = rng.random(200)

        positives = score[truth == 1]
        negatives = score[truth == 0]
        wins = (positives[:, None] > negatives[None, :]).sum()
        ties = (positives[:, None] == negatives[None, :]).sum()
        expected = (wins + 0.5 * ties) / (positives.size * negatives.size)

        assert auroc(truth, score) == pytest.approx(expected)

    def test_single_class_is_undefined(self):
        assert math.isnan(auroc([1, 1, 1], [0.2, 0.5, 0.9]))

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            auroc([], [])

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            auroc([0, 1], [0.5])


class TestAuprc:
    def test_perfect_ranking(self):
        assert auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_all_positive_gives_one(self):
        assert auprc([1, 1, 1], [0.1, 0.5, 0.9]) == pytest.approx(1.0)

    def test_known_value(self):
        # Ranked: pos, neg, pos, neg -> precision 1.0 then 2/3.
        value = auprc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
        assert value == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))

    def test_rare_positive_is_near_base_rate_for_random_scores(self):
        rng = np.random.default_rng(11)
        truth = np.zeros(2000, dtype=int)
        truth[: int(0.08 * 2000)] = 1
        rng.shuffle(truth)
        assert auprc(truth, rng.random(2000)) == pytest.approx(0.08, abs=0.04)

    def test_no_positives_is_undefined(self):
        assert math.isnan(auprc([0, 0], [0.3, 0.7]))


class TestRocCurve:
    def test_starts_at_origin(self):
        fpr, tpr, _ = roc_curve([0, 1], [0.2, 0.8])
        assert (fpr[0], tpr[0]) == (0.0, 0.0)

    def test_ends_at_one(self):
        fpr, tpr, _ = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0

    def test_monotonic(self):
        rng = np.random.default_rng(5)
        truth = rng.integers(0, 2, size=100)
        fpr, tpr, _ = roc_curve(truth, rng.random(100))
        assert np.all(np.diff(fpr) >= 0)
        assert np.all(np.diff(tpr) >= 0)

    def test_requires_both_classes(self):
        with pytest.raises(ValueError):
            roc_curve([1, 1], [0.2, 0.8])


class TestRecallAtFpr:
    def test_perfect_model_reaches_full_recall(self):
        recall, _ = recall_at_fpr([0] * 20 + [1] * 20, [0.1] * 20 + [0.9] * 20, 0.05)
        assert recall == 1.0

    def test_respects_the_fpr_budget(self):
        rng = np.random.default_rng(7)
        truth = np.r_[np.zeros(400, dtype=int), np.ones(40, dtype=int)]
        score = np.r_[rng.normal(0, 1, 400), rng.normal(1.5, 1, 40)]

        recall, threshold = recall_at_fpr(truth, score, 0.05)
        achieved_fpr = (score[truth == 0] >= threshold).mean()
        assert achieved_fpr <= 0.05 + 1e-9
        assert 0.0 < recall < 1.0

    def test_returns_zero_when_budget_is_unreachable(self):
        recall, _ = recall_at_fpr([0, 1], [0.5, 0.5], 0.0)
        assert recall == 0.0


class TestCalibration:
    def test_brier_of_perfect_predictions(self):
        assert brier_score([0, 1], [0.0, 1.0]) == 0.0

    def test_brier_of_worst_predictions(self):
        assert brier_score([0, 1], [1.0, 0.0]) == 1.0

    def test_well_calibrated_predictions_track_the_diagonal(self):
        # Per-bin binomial noise is the limit here: at n/bin ~= 4000 the
        # standard error is about 0.008, so 0.05 leaves ample headroom.
        rng = np.random.default_rng(13)
        size = 40_000
        prob = rng.random(size)
        truth = (rng.random(size) < prob).astype(int)

        predicted, observed, counts = calibration_curve(truth, prob, bins=10)
        assert counts.sum() == size
        assert np.allclose(predicted, observed, atol=0.05)

    def test_only_occupied_bins_returned(self):
        predicted, _, _ = calibration_curve([0, 1], [0.05, 0.06], bins=10)
        assert predicted.size == 1


class TestGroupedBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(19)
        groups = np.repeat([f"repo{i}" for i in range(10)], 40)
        truth = rng.integers(0, 2, size=400)
        score = truth * 0.5 + rng.random(400)

        interval = grouped_bootstrap_ci(truth, score, groups, samples=200)
        assert interval.lower <= interval.point <= interval.upper

    def test_resampling_groups_widens_the_interval_versus_rows(self):
        rng = np.random.default_rng(23)
        # Signal strength differs sharply per repo, so clustering matters.
        truth, score, groups = [], [], []
        for repo in range(8):
            strength = 2.0 if repo % 2 == 0 else 0.0
            labels = rng.integers(0, 2, size=60)
            truth.extend(labels)
            score.extend(labels * strength + rng.random(60))
            groups.extend([f"repo{repo}"] * 60)

        row_ids = [str(i) for i in range(len(truth))]
        clustered = grouped_bootstrap_ci(truth, score, groups, samples=300)
        per_row = grouped_bootstrap_ci(truth, score, row_ids, samples=300)

        assert (clustered.upper - clustered.lower) > (per_row.upper - per_row.lower)

    def test_string_representation(self):
        rng = np.random.default_rng(29)
        groups = np.repeat(["a", "b", "c", "d"], 25)
        truth = rng.integers(0, 2, size=100)
        interval = grouped_bootstrap_ci(truth, rng.random(100), groups, samples=100)
        assert "[" in str(interval)
