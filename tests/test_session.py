"""Session analysis. The measured half must never be confused with the estimate."""

import numpy as np
import pytest

from averta.features import FEATURE_NAMES
from averta.monitor import fit_and_save
from averta.session import MIN_CLUSTERED, WINDOW, analyse, render
from averta.testing import action, observation, session

CUTS = (3, 5, 10, 20, 40)


@pytest.fixture
def scorer(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((300, len(FEATURE_NAMES)))
    y = (rng.random(300) < 0.5).astype(int)
    return fit_and_save(X, y, cut_points=CUTS, path=tmp_path / "m.pkl")


def edits(path: str, at: list[int], length: int) -> list:
    """A session of `length` turns editing `path` at the given turn indices."""
    turns = []
    for i in range(length):
        if i in at:
            turns.append(action(i, path=path))
        else:
            turns.append(observation(i, chars=10))
    return turns


class TestClustering:
    def test_tight_repetition_is_reported(self):
        report = analyse("s", edits("/a.py", [2, 4, 6, 8], 40))
        assert report.repetitions
        assert report.repetitions[0].clustered == 4

    def test_repetition_spread_across_a_long_session_is_ignored(self):
        # The bug this fixes: 27 edits to one file over 1,000 turns is normal
        # iterative work, and ranking by raw count surfaced it above real loops.
        spread = list(range(0, 600, 60))
        report = analyse("s", edits("/a.py", spread, 600))
        assert report.repetitions == ()

    def test_dense_cluster_beats_a_higher_spread_count(self):
        turns = edits("/spread.py", list(range(0, 400, 40)), 400)
        for i in (100, 103, 106, 109, 112):
            turns[i] = action(i, path="/tight.py")

        top = analyse("s", turns).repetitions[0]
        assert top.cluster_start == 100
        assert top.cluster_end == 112
        assert "/tight.py" in top.label or top.kind == "tool call"

    def test_identical_calls_are_not_reported_twice(self):
        # Repeating one call with identical arguments also repeats an edit to
        # the same path, so both collectors fire on the same turns.
        report = analyse("s", edits("/a.py", [2, 4, 6], 30))
        spans = [(r.cluster_start, r.cluster_end) for r in report.repetitions]
        assert len(spans) == len(set(spans))

    def test_below_the_minimum_is_not_reported(self):
        report = analyse("s", edits("/a.py", [1, 3], 30))
        assert report.repetitions == ()

    def test_cluster_bounds_are_reported(self):
        report = analyse("s", edits("/a.py", [5, 7, 9], 40))
        item = report.repetitions[0]
        assert item.cluster_start == 5
        assert item.cluster_end == 9
        assert item.occurrences == 3

    def test_overall_count_can_exceed_the_cluster(self):
        report = analyse("s", edits("/a.py", [1, 3, 5, 200, 400], 500))
        item = report.repetitions[0]
        assert item.clustered == 3
        assert item.occurrences == 5

    def test_window_boundary(self):
        just_inside = [0, 1, WINDOW]
        assert analyse("s", edits("/a.py", just_inside, 200)).repetitions

        just_outside = [0, 1, WINDOW + 5]
        assert analyse("s", edits("/a.py", just_outside, 200)).repetitions == ()


class TestRecurringErrors:
    def test_repeated_error_signature_detected(self):
        turns = [observation(i, error="Boom: x" if i % 2 == 0 else None) for i in range(10)]
        report = analyse("s", turns)
        assert any(r.kind == "error" for r in report.repetitions)

    def test_error_count_excludes_rejections(self):
        turns = [observation(0, error="Boom"), observation(1)]
        report = analyse("s", turns, user_rejections=7)
        assert report.error_turns == 1
        assert report.user_rejections == 7


class TestSinceLastClean:
    def test_counts_back_to_the_last_success(self):
        turns = [observation(0), action(1), observation(2, error="E")]
        assert analyse("s", turns).turns_since_clean == 2

    def test_whole_session_when_nothing_succeeded(self):
        turns = [observation(i, error="E") for i in range(4)]
        assert analyse("s", turns).turns_since_clean == 4


class TestTrajectory:
    def test_no_trajectory_without_a_model(self):
        assert analyse("s", session(50)).trajectory == ()

    def test_scores_at_each_reachable_cut(self, scorer):
        report = analyse("s", session(50), scorer=scorer)
        assert [p.turn for p in report.trajectory] == [5, 10, 20, 40]

    def test_stops_at_the_largest_trained_cut(self, scorer):
        report = analyse("s", session(900), scorer=scorer)
        assert report.scored_through == 40
        assert report.trajectory[-1].turn == 40

    def test_short_session_yields_fewer_points(self, scorer):
        report = analyse("s", session(12), scorer=scorer)
        assert [p.turn for p in report.trajectory] == [5, 10]

    def test_direction_is_unknown_with_one_point(self, scorer):
        assert analyse("s", session(6), scorer=scorer).risk_direction == "unknown"


class TestRender:
    def test_separates_measured_from_estimated(self, scorer):
        text = render(analyse("s", session(60), scorer=scorer), base_rate=0.88)
        assert "MEASURED — exact, no model involved" in text
        assert "ESTIMATED — model did not clear its gate" in text
        assert text.index("MEASURED") < text.index("ESTIMATED")

    def test_shows_the_base_rate_for_comparison(self, scorer):
        text = render(analyse("s", session(60), scorer=scorer), base_rate=0.88)
        assert "88.0%" in text
        assert "not against zero" in text

    def test_states_when_nothing_repeated(self):
        text = render(analyse("s", [observation(i) for i in range(10)]))
        assert "no clustered repetition" in text

    def test_omits_the_estimate_entirely_without_a_model(self):
        text = render(analyse("s", session(60)))
        assert "ESTIMATED" not in text
        assert "MEASURED" in text

    def test_explains_why_the_curve_stops(self, scorer):
        text = render(analyse("s", session(500), scorer=scorer))
        assert "stops at turn 40" in text

    def test_minimum_cluster_size_is_stated(self):
        text = render(analyse("s", [observation(i) for i in range(10)]))
        assert str(MIN_CLUSTERED) in text
