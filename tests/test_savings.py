import numpy as np

from averta.savings import CHARS_PER_TOKEN, render, simulate


def estimates(n: int, remaining: float = 4000.0, total: float = 8000.0):
    return {f"s{i}": (remaining, total) for i in range(n)}


class TestSimulate:
    def test_perfect_model_at_low_threshold_flags_all_failures(self):
        ids = np.array([f"s{i}" for i in range(10)])
        y = np.array([1] * 5 + [0] * 5)
        p = np.array([0.9] * 5 + [0.1] * 5)

        point = simulate(ids, y, p, estimates(10), thresholds=np.array([0.5]))[0]
        assert point.recall == 1.0
        assert point.false_positive_rate == 0.0
        assert point.successes_terminated == 0

    def test_successes_terminated_are_counted(self):
        ids = np.array(["s0", "s1"])
        y = np.array([1, 0])
        p = np.array([0.9, 0.9])

        point = simulate(ids, y, p, estimates(2), thresholds=np.array([0.5]))[0]
        assert point.successes_terminated == 1
        assert point.false_positive_rate == 1.0

    def test_savings_exclude_wrongly_flagged_successes(self):
        # Only the true positive contributes tokens saved.
        ids = np.array(["s0", "s1"])
        y = np.array([1, 0])
        p = np.array([0.9, 0.9])

        point = simulate(ids, y, p, estimates(2, remaining=1000.0), thresholds=np.array([0.5]))[0]
        assert point.estimated_tokens_saved == 1000

    def test_raising_the_threshold_lowers_recall_and_savings(self):
        rng = np.random.default_rng(0)
        ids = np.array([f"s{i}" for i in range(200)])
        y = (rng.random(200) < 0.5).astype(int)
        p = np.clip(y * 0.4 + rng.random(200) * 0.5, 0, 1)

        points = simulate(ids, y, p, estimates(200), thresholds=np.array([0.3, 0.6, 0.9]))
        assert points[0].recall >= points[1].recall >= points[2].recall
        assert (
            points[0].estimated_tokens_saved
            >= points[1].estimated_tokens_saved
            >= points[2].estimated_tokens_saved
        )

    def test_sessions_missing_estimates_contribute_nothing(self):
        ids = np.array(["unknown"])
        point = simulate(ids, np.array([1]), np.array([0.9]), {}, thresholds=np.array([0.5]))[0]
        assert point.estimated_tokens_saved == 0

    def test_savings_rate_handles_zero_total(self):
        ids = np.array(["s0"])
        point = simulate(
            ids, np.array([1]), np.array([0.9]), {"s0": (0.0, 0.0)}, thresholds=np.array([0.5])
        )[0]
        assert point.savings_rate == 0.0

    def test_no_positives_gives_zero_recall(self):
        ids = np.array(["s0", "s1"])
        point = simulate(
            ids, np.array([0, 0]), np.array([0.9, 0.9]), estimates(2),
            thresholds=np.array([0.5]),
        )[0]
        assert point.recall == 0.0


class TestRender:
    def test_labels_tokens_as_estimated(self):
        ids = np.array(["s0"])
        points = simulate(ids, np.array([1]), np.array([0.9]), estimates(1))
        text = render(points)
        assert "ESTIMATED" in text
        assert f"{CHARS_PER_TOKEN:.0f} chars/token" in text

    def test_states_that_harm_is_not_netted_off(self):
        ids = np.array(["s0"])
        points = simulate(ids, np.array([1]), np.array([0.9]), estimates(1))
        assert "never netted off" in render(points)
