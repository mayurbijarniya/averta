from averta.features.sequence import (
    action_bigram_repeat_max,
    action_compression_ratio,
    action_trigram_repeat_max,
    alternation_count,
    distinct_action_ratio,
    edit_then_error_rate,
    longest_identical_run,
    mean_return_distance,
    novelty_rate_recent,
    repeat_acceleration,
)
from tests.factories import action, observation


def actions(paths: list[str]) -> list:
    return [action(i, path=f"/workspace/{p}.py") for i, p in enumerate(paths)]


class TestNgramRepeats:
    def test_bigram_repeat_detected(self):
        # a,b,a,b contains the pair (a,b) twice.
        assert action_bigram_repeat_max(actions(["a", "b", "a", "b"])) == 2

    def test_bigram_without_repeats(self):
        assert action_bigram_repeat_max(actions(["a", "b", "c", "d"])) == 1

    def test_trigram_repeat_detected(self):
        assert action_trigram_repeat_max(actions(["a", "b", "c", "a", "b", "c"])) == 2

    def test_too_short_for_ngram(self):
        assert action_bigram_repeat_max(actions(["a"])) == 0.0
        assert action_trigram_repeat_max(actions(["a", "b"])) == 0.0

    def test_empty_prefix(self):
        assert action_bigram_repeat_max([]) == 0.0


class TestRuns:
    def test_consecutive_identical_calls(self):
        assert longest_identical_run(actions(["a", "a", "a", "b"])) == 3

    def test_non_consecutive_repeats_do_not_count(self):
        assert longest_identical_run(actions(["a", "b", "a", "b"])) == 1

    def test_single_action(self):
        assert longest_identical_run(actions(["a"])) == 1

    def test_empty(self):
        assert longest_identical_run([]) == 0.0


class TestAlternation:
    def test_aba_detected(self):
        assert alternation_count(actions(["a", "b", "a"])) == 1

    def test_repeated_alternation(self):
        assert alternation_count(actions(["a", "b", "a", "b", "a"])) == 3

    def test_identical_run_is_not_alternation(self):
        assert alternation_count(actions(["a", "a", "a"])) == 0

    def test_distinct_sequence(self):
        assert alternation_count(actions(["a", "b", "c"])) == 0


class TestDiversity:
    def test_all_distinct(self):
        assert distinct_action_ratio(actions(["a", "b", "c"])) == 1.0

    def test_all_identical(self):
        assert distinct_action_ratio(actions(["a", "a", "a", "a"])) == 0.25

    def test_empty(self):
        assert distinct_action_ratio([]) == 0.0

    def test_novelty_is_one_for_fresh_actions(self):
        assert novelty_rate_recent(actions(["a", "b", "c", "d"])) == 1.0

    def test_novelty_falls_when_revisiting(self):
        assert novelty_rate_recent(actions(["a", "b", "a", "b"])) < 1.0

    def test_novelty_of_single_action(self):
        assert novelty_rate_recent(actions(["a"])) == 1.0


class TestReturnDistance:
    def test_adjacent_repeat_has_distance_one(self):
        assert mean_return_distance(actions(["a", "a"])) == 1.0

    def test_distant_repeat(self):
        assert mean_return_distance(actions(["a", "b", "c", "a"])) == 3.0

    def test_no_repeats(self):
        assert mean_return_distance(actions(["a", "b", "c"])) == 0.0


class TestCompression:
    def test_repetitive_compresses_better_than_varied(self):
        repetitive = action_compression_ratio(actions(["a"] * 60))
        varied = action_compression_ratio(actions([f"f{i}" for i in range(60)]))
        assert repetitive < varied

    def test_short_sequence_returns_one(self):
        assert action_compression_ratio(actions(["a"])) == 1.0

    def test_empty_returns_one(self):
        assert action_compression_ratio([]) == 1.0


class TestEditThenError:
    def test_edit_followed_by_error(self):
        prefix = [action(0), observation(1, error="E: boom")]
        assert edit_then_error_rate(prefix) == 1.0

    def test_edit_followed_by_success(self):
        prefix = [action(0), observation(1)]
        assert edit_then_error_rate(prefix) == 0.0

    def test_mixed(self):
        prefix = [
            action(0, path="/workspace/a.py"),
            observation(1, error="E: boom"),
            action(2, path="/workspace/b.py"),
            observation(3),
        ]
        assert edit_then_error_rate(prefix) == 0.5

    def test_no_edits(self):
        assert edit_then_error_rate([observation(0)]) == 0.0

    def test_edit_with_no_following_observation(self):
        assert edit_then_error_rate([action(0)]) == 0.0


class TestRepeatAcceleration:
    def test_repeats_concentrated_late_exceed_one(self):
        prefix = actions(["a", "b", "c", "d", "e", "e", "e", "e"])
        assert repeat_acceleration(prefix) > 1.0

    def test_repeats_concentrated_early_stay_below_one(self):
        prefix = actions(["a", "a", "a", "a", "b", "c", "d", "e"])
        assert repeat_acceleration(prefix) < 1.0

    def test_too_short(self):
        assert repeat_acceleration(actions(["a", "b"])) == 0.0

    def test_no_repeats_anywhere(self):
        assert repeat_acceleration(actions(["a", "b", "c", "d"])) == 0.0
