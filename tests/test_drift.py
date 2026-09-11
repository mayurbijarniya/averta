"""Cross-scaffold feature comparison, and the vocabulary bug it exposed."""

import json

import numpy as np
import pytest

from averta.drift import MIN_SAMPLES_FOR_INFERENCE, compare
from averta.features.sequence import action_compression_ratio
from averta.features.view import TurnView, to_views
from averta.normalize import digest
from averta.testing import action

NAMES = ("a", "b")


def claude_edit_turn(path: str = "/repo/main.py", tool: str = "Edit") -> TurnView:
    payload = json.dumps({"file_path": path, "old_string": "x", "new_string": "y"})
    return TurnView(
        turn_index=0,
        role="assistant",
        step_index=0,
        tool_name=tool,
        tool_input=payload,
        tool_input_hash=digest(payload),
        tool_input_bad=False,
        n_tool_calls=1,
        content_chars=10,
        is_error=False,
        error_signature=None,
    )


class TestToolVocabularies:
    """Both scaffolds express editing differently; both must be recognised."""

    def test_openhands_editor_is_recognised(self):
        assert action(0, path="/workspace/a.py").edited_path == "/workspace/a.py"

    @pytest.mark.parametrize("tool", ["Edit", "Write", "MultiEdit", "NotebookEdit"])
    def test_claude_code_edit_tools_are_recognised(self, tool):
        assert claude_edit_turn(tool=tool).edited_path == "/repo/main.py"

    def test_unknown_tool_is_not_an_edit(self):
        assert claude_edit_turn(tool="Read").edited_path is None

    def test_bash_is_not_an_edit(self):
        turn = claude_edit_turn(tool="Bash")
        assert turn.edited_path is None

    def test_notebook_path_key_is_read(self):
        payload = json.dumps({"notebook_path": "/repo/nb.ipynb"})
        turn = claude_edit_turn()._replace(
            tool_name="NotebookEdit", tool_input=payload
        )
        assert turn.edited_path == "/repo/nb.ipynb"

    def test_missing_path_yields_none(self):
        turn = claude_edit_turn()._replace(tool_input=json.dumps({"old_string": "x"}))
        assert turn.edited_path is None


class TestCompressionRatioBounds:
    def test_ratio_never_exceeds_one(self):
        # zlib's header made short sequences report 1.85 before raw deflate.
        for length in range(2, 40):
            prefix = [action(i, path=f"/w/f{i}.py") for i in range(length)]
            assert action_compression_ratio(prefix) <= 1.0

    def test_repetitive_still_compresses_further(self):
        repetitive = action_compression_ratio([action(i, path="/w/same.py") for i in range(80)])
        varied = action_compression_ratio([action(i, path=f"/w/f{i}.py") for i in range(80)])
        assert repetitive < varied


class TestCompare:
    def test_reports_sample_sizes(self):
        report = compare(NAMES, np.zeros((100, 2)), np.zeros((5, 2)))
        assert report.n_corpus == 100
        assert report.n_local == 5

    def test_flags_underpowered_samples(self):
        assert compare(NAMES, np.zeros((100, 2)), np.zeros((5, 2))).underpowered
        big = np.zeros((MIN_SAMPLES_FOR_INFERENCE, 2))
        assert not compare(NAMES, np.zeros((100, 2)), big).underpowered

    def test_identical_distributions_show_no_shift(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (200, 2))
        report = compare(NAMES, data, data)
        assert all(abs(s.standardized_difference) < 1e-9 for s in report.shifts)
        assert all(s.magnitude == "negligible" for s in report.shifts)

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(1)
        corpus = rng.normal(0, 1, (500, 2))
        local = rng.normal(3, 1, (40, 2))
        report = compare(NAMES, corpus, local)
        assert all(s.standardized_difference > 2 for s in report.shifts)
        assert all(s.magnitude == "large" for s in report.shifts)

    def test_out_of_range_values_are_flagged(self):
        corpus = np.zeros((50, 2))
        local = np.full((3, 2), 99.0)
        report = compare(NAMES, corpus, local)
        assert len(report.out_of_range) == 2

    def test_in_range_values_are_not_flagged(self):
        rng = np.random.default_rng(2)
        corpus = rng.normal(0, 1, (500, 2))
        local = np.zeros((3, 2))
        assert compare(NAMES, corpus, local).out_of_range == ()

    def test_zero_variance_feature_does_not_divide_by_zero(self):
        report = compare(NAMES, np.zeros((50, 2)), np.ones((3, 2)))
        assert all(s.standardized_difference == 0.0 for s in report.shifts)

    def test_feature_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="feature count"):
            compare(NAMES, np.zeros((10, 3)), np.zeros((3, 3)))

    def test_render_warns_when_underpowered(self):
        text = compare(NAMES, np.zeros((100, 2)), np.zeros((3, 2))).render()
        assert "UNDERPOWERED" in text
        assert "not an estimate" in text


class TestEndToEnd:
    def test_claude_style_turns_register_edits(self):
        """The regression that started this: edits must not silently vanish."""
        rows = [
            {
                "turn_index": i,
                "role": "assistant",
                "step_index": i,
                "tool_name": "Edit",
                "tool_input": json.dumps({"file_path": f"/repo/f{i}.py"}),
                "tool_input_hash": f"h{i}",
                "tool_input_bad": False,
                "n_tool_calls": 1,
                "content_chars": 10,
                "is_error": False,
                "error_signature": None,
            }
            for i in range(4)
        ]
        views = to_views(rows)
        assert sum(1 for v in views if v.edited_path) == 4
