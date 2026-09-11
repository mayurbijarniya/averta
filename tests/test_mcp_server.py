"""The MCP surface must never report a number without its caveats."""

import numpy as np
import pytest

from averta import mcp_server
from averta.features import FEATURE_NAMES
from averta.monitor import fit_and_save
from tests.test_monitor import assistant, tool_result, write_transcript

CUTS = (3, 5, 10, 20, 40)


@pytest.fixture
def local_session(tmp_path, monkeypatch):
    records = []
    for i in range(24):
        records.append(assistant(f"step {i}", tool="Bash", tool_input={"command": "ls"}))
        records.append(tool_result("output", is_error=i % 5 == 0))
    path = write_transcript(tmp_path, records)

    monkeypatch.setattr(mcp_server, "discover_transcripts", lambda *a, **k: [path])

    rng = np.random.default_rng(0)
    X = rng.random((200, len(FEATURE_NAMES)))
    y = (rng.random(200) < 0.5).astype(int)
    model_path = tmp_path / "m.pkl"
    fit_and_save(X, y, cut_points=CUTS, path=model_path)
    monkeypatch.setattr(mcp_server, "DEFAULT_MODEL_PATH", model_path)
    return path


class TestGetSessionRisk:
    def test_returns_a_probability(self, local_session):
        result = mcp_server.get_session_risk()
        assert 0.0 <= result["failure_probability"] <= 1.0

    def test_always_includes_caveats(self, local_session):
        caveats = mcp_server.get_session_risk()["caveats"]
        assert any("pre-registered" in c for c in caveats)
        assert any("unmeasured" in c for c in caveats)

    def test_reports_which_turn_it_scored(self, local_session):
        assert mcp_server.get_session_risk()["scored_at_turn"] in CUTS

    def test_unknown_session_raises(self, local_session):
        with pytest.raises(ValueError, match="no transcript matching"):
            mcp_server.get_session_risk(session="nonexistent")


class TestGetRepeatedFailures:
    def test_groups_recurring_errors(self, local_session):
        result = mcp_server.get_repeated_failures()
        assert result["distinct_errors"] >= 1

    def test_reissued_calls_are_distinguishable(self, local_session):
        for entry in mcp_server.get_repeated_failures()["reissued_calls"]:
            assert entry["arguments_id"]
            assert entry["occurrences"] > 1

    def test_states_that_counts_are_measured(self, local_session):
        assert "measured, not predicted" in mcp_server.get_repeated_failures()["note"]


class TestShouldIRestart:
    def test_recommends_continue_below_threshold(self, local_session):
        result = mcp_server.should_i_restart(threshold=0.99)
        assert result["recommendation"] == "continue"

    def test_recommends_restart_above_threshold(self, local_session):
        result = mcp_server.should_i_restart(threshold=0.0)
        assert result["recommendation"] == "consider restarting"

    def test_warns_that_continue_is_weak_evidence(self, local_session):
        caveats = mcp_server.should_i_restart()["caveats"]
        assert any("weak evidence" in c for c in caveats)

    def test_default_threshold_is_conservative(self, local_session):
        assert mcp_server.should_i_restart()["threshold"] >= 0.7


class TestListSessions:
    def test_lists_local_sessions(self, local_session):
        result = mcp_server.list_sessions()
        assert result["count"] == 1
        assert result["sessions"][0]["turns"] > 0

    def test_no_transcripts_raises_on_risk(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "discover_transcripts", lambda *a, **k: [])
        with pytest.raises(ValueError, match="no transcripts found"):
            mcp_server.get_session_risk()
