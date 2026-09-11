from averta.features.extractors import (
    chars_growth_ratio,
    error_rate,
    has_finished,
    max_call_repeat,
    max_error_repeat,
    max_file_edit_repeat,
    n_distinct_errors,
    n_distinct_tools,
    n_files_touched,
    n_malformed_tool_inputs,
    n_repeated_calls,
    n_test_commands,
    recent_call_repeat_rate,
    tool_entropy,
    turns_since_clean_observation,
    turns_since_error,
)
from tests.factories import action, bash, observation


class TestErrorFeatures:
    def test_max_error_repeat_counts_the_same_signature(self):
        prefix = [
            observation(0, error="ValueError: nope"),
            observation(1, error="ValueError: nope"),
            observation(2, error="TypeError: other"),
            observation(3, error="ValueError: nope"),
        ]
        assert max_error_repeat(prefix) == 3
        assert n_distinct_errors(prefix) == 2

    def test_no_errors_yields_zero(self):
        assert max_error_repeat([observation(0), observation(1)]) == 0

    def test_error_rate_is_over_observations_not_all_turns(self):
        prefix = [action(0), observation(1, error="E: x"), action(2), observation(3)]
        assert error_rate(prefix) == 0.5

    def test_error_rate_with_no_observations(self):
        assert error_rate([action(0), action(1)]) == 0.0

    def test_turns_since_error_counts_back_from_the_end(self):
        prefix = [observation(0, error="E: x"), action(1), observation(2)]
        assert turns_since_error(prefix) == 2

    def test_turns_since_error_without_any_error(self):
        prefix = [action(0), observation(1)]
        assert turns_since_error(prefix) == 2

    def test_turns_since_clean_observation(self):
        prefix = [
            observation(0),
            action(1),
            observation(2, error="E: x"),
            action(3),
        ]
        assert turns_since_clean_observation(prefix) == 3


class TestRepetitionFeatures:
    def test_identical_calls_are_counted(self):
        prefix = [action(i, path="/workspace/same.py") for i in range(4)]
        assert max_call_repeat(prefix) == 4
        assert n_repeated_calls(prefix) == 3

    def test_distinct_calls_are_not_counted(self):
        prefix = [action(i, path=f"/workspace/f{i}.py") for i in range(4)]
        assert max_call_repeat(prefix) == 1
        assert n_repeated_calls(prefix) == 0

    def test_recent_repeat_rate_detects_returning_to_an_old_call(self):
        prefix = [action(i, path=f"/workspace/f{i}.py") for i in range(6)]
        prefix.append(action(6, path="/workspace/f0.py"))
        assert recent_call_repeat_rate(prefix) > 0

    def test_recent_repeat_rate_needs_history(self):
        assert recent_call_repeat_rate([action(0)]) == 0.0


class TestFileFeatures:
    def test_distinct_paths_counted_once(self):
        prefix = [
            action(0, path="/workspace/a.py"),
            action(1, path="/workspace/a.py"),
            action(2, path="/workspace/b.py"),
        ]
        assert n_files_touched(prefix) == 2
        assert max_file_edit_repeat(prefix) == 2

    def test_non_edit_commands_do_not_count_as_edits(self):
        prefix = [action(0, command="view", path="/workspace/a.py")]
        assert n_files_touched(prefix) == 0

    def test_malformed_arguments_are_counted(self):
        prefix = [action(0), action(1, malformed=True), action(2, malformed=True)]
        assert n_malformed_tool_inputs(prefix) == 2

    def test_malformed_arguments_yield_no_path(self):
        prefix = [action(0, malformed=True)]
        assert n_files_touched(prefix) == 0


class TestToolFeatures:
    def test_entropy_is_zero_for_a_single_tool(self):
        assert tool_entropy([action(i) for i in range(5)]) == 0.0

    def test_entropy_is_one_bit_for_an_even_split(self):
        prefix = [action(0), bash(1, "ls"), action(2), bash(3, "pwd")]
        assert tool_entropy(prefix) == 1.0

    def test_entropy_of_empty_prefix(self):
        assert tool_entropy([]) == 0.0

    def test_distinct_tools(self):
        prefix = [action(0), bash(1, "ls"), action(2)]
        assert n_distinct_tools(prefix) == 2

    def test_test_commands_detected(self):
        prefix = [bash(0, "python -m pytest tests/"), bash(1, "ls -la")]
        assert n_test_commands(prefix) == 1

    def test_finish_tool_detected(self):
        assert has_finished([action(0, tool="finish")]) == 1.0
        assert has_finished([action(0)]) == 0.0


class TestVolumeFeatures:
    def test_growth_ratio_is_one_for_flat_output(self):
        prefix = [observation(i, chars=100) for i in range(10)]
        assert chars_growth_ratio(prefix) == 1.0

    def test_growth_ratio_exceeds_one_when_output_accelerates(self):
        prefix = [observation(i, chars=10) for i in range(10)]
        prefix += [observation(i, chars=500) for i in range(10, 15)]
        assert chars_growth_ratio(prefix) > 1.0

    def test_growth_ratio_handles_zero_content(self):
        assert chars_growth_ratio([observation(0, chars=0)]) == 0.0

    def test_growth_ratio_of_empty_prefix(self):
        assert chars_growth_ratio([]) == 0.0
