"""Which session a command or an agent is asking about.

The MCP server is never told who is calling it, and a report on the wrong
session reads exactly like a report on the right one, so the choice has to be
right by default and stated every time.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from averta import mcp_server
from averta.adapters import (
    SelectionError,
    discover_transcripts,
    project_transcripts,
    select_transcript,
    transcript_cwd,
    transcript_root,
)
from averta.cli import app
from averta.monitor import DEFAULT_MODEL_PATH, Scorer
from averta.testing import claude_assistant_record as assistant
from averta.testing import claude_tool_result_record as tool_result

runner = CliRunner()


def write(root: Path, cwd: Path, session_id: str, mtime: float, records=None) -> Path:
    """A transcript for a session started in `cwd`, last written at `mtime`."""
    folder = root / ("-" + str(cwd).strip("/").replace("/", "-"))
    folder.mkdir(exist_ok=True)
    path = folder / f"{session_id}.jsonl"
    records = records or [assistant("hi", cwd=str(cwd))]
    path.write_text("\n".join(json.dumps(r) for r in records))
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def projects(tmp_path):
    a = tmp_path / "alpha"
    b = tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    return a, b


class TestRoot:
    def test_follows_claude_config_dir(self, isolated_transcripts):
        assert transcript_root() == isolated_transcripts

    def test_defaults_to_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert transcript_root() == tmp_path / ".claude" / "projects"


class TestTranscriptCwd:
    def test_reads_first_record_carrying_cwd(self, isolated_transcripts, projects):
        a, b = projects
        path = write(
            isolated_transcripts,
            a,
            "s",
            1000,
            [{"type": "mode", "mode": "default"}, assistant(cwd=str(a)), assistant(cwd=str(b))],
        )
        assert transcript_cwd(path) == str(a)

    def test_none_when_absent(self, isolated_transcripts, projects):
        path = write(isolated_transcripts, projects[0], "s", 1000, [{"type": "mode"}])
        assert transcript_cwd(path) is None


class TestProjectSelection:
    def test_prefers_this_project_over_a_newer_one_elsewhere(
        self, isolated_transcripts, projects
    ):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other", 2000)
        selection = select_transcript(here=a)
        assert selection.session_id == "mine"
        assert selection.selected_by == "project"
        assert selection.warnings == ()

    def test_most_recent_within_the_project(self, isolated_transcripts, projects):
        a, _ = projects
        write(isolated_transcripts, a, "old", 1000)
        write(isolated_transcripts, a, "new", 5000)
        assert select_transcript(here=a).session_id == "new"

    def test_working_directory_is_the_default_project(
        self, isolated_transcripts, projects, monkeypatch
    ):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other", 2000)
        monkeypatch.chdir(a)
        assert select_transcript().session_id == "mine"

    def test_averta_project_overrides_the_working_directory(
        self, isolated_transcripts, projects, monkeypatch
    ):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other", 2000)
        monkeypatch.chdir(b)
        monkeypatch.setenv("AVERTA_PROJECT", str(a))
        assert select_transcript().session_id == "mine"

    def test_subdirectory_resolves_to_the_enclosing_project(
        self, isolated_transcripts, projects
    ):
        a, b = projects
        sub = a / "src" / "pkg"
        sub.mkdir(parents=True)
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other", 2000)
        selection = select_transcript(here=sub)
        assert selection.session_id == "mine"
        assert selection.selected_by == "parent"
        assert str(a) in selection.describe()

    def test_exact_match_beats_a_fresher_enclosing_session(
        self, isolated_transcripts, projects
    ):
        a, _ = projects
        sub = a / "sub"
        sub.mkdir()
        write(isolated_transcripts, sub, "exact", 1000)
        write(isolated_transcripts, a, "enclosing", 9000)
        assert select_transcript(here=sub).session_id == "exact"

    def test_workspace_folder_does_not_leak_into_a_project(
        self, isolated_transcripts, tmp_path
    ):
        # Sessions started in a folder holding many projects belong to none
        # of them once the project has sessions of its own.
        workspace = tmp_path / "code"
        project = workspace / "proj"
        (project / "src").mkdir(parents=True)
        write(isolated_transcripts, project, "mine", 1000)
        write(isolated_transcripts, workspace, "workspace", 9000)
        listed = project_transcripts(project / "src", discover_transcripts())
        assert [p.stem for p, _ in listed] == ["mine"]

    def test_sessions_below_here_are_not_claimed(self, isolated_transcripts, projects):
        # Run from a parent directory, a session in a child is a different
        # project; claiming it would make a home directory match everything.
        a, _ = projects
        write(isolated_transcripts, a, "child", 1000)
        assert project_transcripts(a.parent, discover_transcripts()) == []

    def test_home_directory_does_not_enclose_projects(
        self, isolated_transcripts, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        project = home / "code" / "proj"
        project.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        write(isolated_transcripts, home, "started-in-home", 1000)
        selection = select_transcript(here=project)
        assert selection.selected_by == "fallback"

    def test_symlinked_path_is_the_same_project(self, isolated_transcripts, projects, tmp_path):
        a, b = projects
        link = tmp_path / "link"
        link.symlink_to(a)
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other", 2000)
        assert select_transcript(here=link).session_id == "mine"

    @pytest.mark.skipif(sys.platform != "darwin", reason="case-insensitive filesystem")
    def test_case_variant_path_is_the_same_project(self, isolated_transcripts, tmp_path):
        project = tmp_path / "MyProj"
        project.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        write(isolated_transcripts, project, "mine", 1000)
        write(isolated_transcripts, other, "other", 2000)
        assert select_transcript(here=tmp_path / "myproj").session_id == "mine"

    def test_concurrent_sessions_in_one_project_are_flagged(
        self, isolated_transcripts, projects
    ):
        a, _ = projects
        write(isolated_transcripts, a, "first", 1000)
        write(isolated_transcripts, a, "second", 1010)
        selection = select_transcript(here=a)
        assert selection.session_id == "second"
        assert any("AVERTA_SESSION" in w for w in selection.warnings)

    def test_stale_sibling_is_not_flagged(self, isolated_transcripts, projects):
        a, _ = projects
        write(isolated_transcripts, a, "yesterday", 1000)
        write(isolated_transcripts, a, "today", 1000 + 86400)
        assert select_transcript(here=a).warnings == ()


class TestFallback:
    def test_falls_back_and_says_so(self, isolated_transcripts, projects, tmp_path):
        a, _ = projects
        write(isolated_transcripts, a, "elsewhere", 1000)
        nowhere = tmp_path / "nowhere"
        nowhere.mkdir()
        selection = select_transcript(here=nowhere)
        assert selection.session_id == "elsewhere"
        assert selection.selected_by == "fallback"
        assert "may not be the one you meant" in selection.warnings[0]

    def test_no_transcripts_at_all(self, tmp_path):
        with pytest.raises(SelectionError, match="no transcripts found"):
            select_transcript(here=tmp_path)


class TestExplicitSession:
    def test_id_matches_across_projects(self, isolated_transcripts, projects):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other-1234", 2000)
        selection = select_transcript("other", here=a)
        assert selection.session_id == "other-1234"
        assert selection.selected_by == "explicit"

    def test_ambiguous_prefix_is_refused(self, isolated_transcripts, projects):
        a, _ = projects
        write(isolated_transcripts, a, "abc-1", 1000)
        write(isolated_transcripts, a, "abc-2", 2000)
        with pytest.raises(SelectionError, match="matches 2 transcripts"):
            select_transcript("abc", here=a)

    def test_exact_id_wins_over_a_longer_prefix_match(self, isolated_transcripts, projects):
        a, _ = projects
        write(isolated_transcripts, a, "abc", 1000)
        write(isolated_transcripts, a, "abcdef", 2000)
        assert select_transcript("abc", here=a).session_id == "abc"

    def test_unknown_id(self, isolated_transcripts, projects):
        write(isolated_transcripts, projects[0], "mine", 1000)
        with pytest.raises(SelectionError, match="no transcript matching"):
            select_transcript("zzz", here=projects[0])

    def test_env_pins_a_session(self, isolated_transcripts, projects, monkeypatch):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "pinned", 500)
        monkeypatch.setenv("AVERTA_SESSION", "pinned")
        selection = select_transcript(here=a)
        assert selection.session_id == "pinned"
        assert selection.selected_by == "env"

    def test_argument_beats_env(self, isolated_transcripts, projects, monkeypatch):
        a, _ = projects
        write(isolated_transcripts, a, "one", 1000)
        write(isolated_transcripts, a, "two", 2000)
        monkeypatch.setenv("AVERTA_SESSION", "two")
        assert select_transcript("one", here=a).session_id == "one"


class TestShippedModel:
    def test_lives_inside_the_package(self):
        import averta

        assert DEFAULT_MODEL_PATH.parent == Path(averta.__file__).resolve().parent
        assert DEFAULT_MODEL_PATH.is_file()

    def test_loads_from_any_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        scorer = Scorer.load()
        assert scorer.feature_names
        assert scorer.base_rate is not None

    def test_is_a_complete_scorer_payload(self):
        with DEFAULT_MODEL_PATH.open("rb") as handle:
            payload = pickle.load(handle)
        assert {"model", "feature_names", "cut_points", "calibrator", "base_rate"} <= set(
            payload
        )


def stuck_session(cwd: Path, repeats: int = 6) -> list[dict]:
    """The same failing command, issued `repeats` times, with known usage."""
    usage = {"input_tokens": 10, "output_tokens": 100, "cache_read_input_tokens": 1000}
    records = [{"type": "user", "cwd": str(cwd), "message": {"role": "user", "content": "go"}}]
    for _ in range(repeats):
        records.append(
            assistant(
                "", tool="Bash", tool_input={"command": "pytest tests/"}, usage=usage,
                cwd=str(cwd),
            )
        )
        records.append(
            tool_result("ModuleNotFoundError: No module named 'foo'", is_error=True,
                        cwd=str(cwd))
        )
    return records


class TestCommandLine:
    def test_explain_reads_this_project_and_names_the_transcript(
        self, isolated_transcripts, projects, monkeypatch
    ):
        a, b = projects
        mine = write(isolated_transcripts, a, "mine", 1000, stuck_session(a))
        write(isolated_transcripts, b, "other", 2000)
        monkeypatch.chdir(a)

        result = runner.invoke(app, ["explain"])
        assert result.exit_code == 0, result.output
        assert f"transcript: {mine}" in result.output
        assert "most recent session started in" in result.output
        assert "session mine" in result.output

    def test_explain_measured_counts_are_exact(
        self, isolated_transcripts, projects, monkeypatch
    ):
        a, _ = projects
        write(isolated_transcripts, a, "mine", 1000, stuck_session(a, repeats=6))
        monkeypatch.chdir(a)

        out = runner.invoke(app, ["explain"]).output
        assert "agent errors            6" in out
        assert "6x  error" in out
        assert "6x  tool call" in out
        assert "No module named 'foo'" in out
        assert f"output              {600:>14,}" in out
        assert f"input, uncached     {60:>14,}" in out
        assert f"cache read          {6000:>14,}" in out

    def test_explain_uses_the_shipped_model_from_anywhere(
        self, isolated_transcripts, projects, monkeypatch
    ):
        a, _ = projects
        write(isolated_transcripts, a, "mine", 1000, stuck_session(a))
        monkeypatch.chdir(a)
        out = runner.invoke(app, ["explain"]).output
        assert "ESTIMATED" in out
        assert "no model at" not in out

    def test_explain_says_when_the_model_is_missing(
        self, isolated_transcripts, projects, monkeypatch, tmp_path
    ):
        a, _ = projects
        write(isolated_transcripts, a, "mine", 1000, stuck_session(a))
        monkeypatch.chdir(a)
        result = runner.invoke(app, ["explain", "--model-path", str(tmp_path / "gone.pkl")])
        assert result.exit_code == 0
        assert "no model at" in result.output
        assert "MEASURED" in result.output

    def test_explain_reports_fallback(self, isolated_transcripts, projects, monkeypatch, tmp_path):
        a, _ = projects
        write(isolated_transcripts, a, "elsewhere", 1000, stuck_session(a))
        nowhere = tmp_path / "nowhere"
        nowhere.mkdir()
        monkeypatch.chdir(nowhere)
        assert "WARNING" in runner.invoke(app, ["explain"]).output

    def test_score_follows_the_same_selection(self, isolated_transcripts, projects, monkeypatch):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000, stuck_session(a))
        write(isolated_transcripts, b, "other", 2000, stuck_session(b))
        monkeypatch.chdir(a)
        result = runner.invoke(app, ["score"])
        assert result.exit_code == 0, result.output
        assert "session mine" in result.output

    def test_ambiguous_id_is_a_usage_error(self, isolated_transcripts, projects, monkeypatch):
        a, _ = projects
        write(isolated_transcripts, a, "abc-1", 1000)
        write(isolated_transcripts, a, "abc-2", 2000)
        monkeypatch.chdir(a)
        result = runner.invoke(app, ["explain", "abc"])
        assert result.exit_code != 0
        assert "matches 2 transcripts" in result.output

    def test_sessions_lists_this_project_by_default(
        self, isolated_transcripts, projects, monkeypatch
    ):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000)
        write(isolated_transcripts, b, "other", 2000)
        monkeypatch.chdir(a)
        scoped = runner.invoke(app, ["sessions"]).output
        assert "mine" in scoped
        assert "other" not in scoped
        everything = runner.invoke(app, ["sessions", "--all"]).output
        assert "mine" in everything and "other" in everything

    def test_sessions_falls_back_to_every_project(
        self, isolated_transcripts, projects, monkeypatch, tmp_path
    ):
        a, _ = projects
        write(isolated_transcripts, a, "mine", 1000)
        nowhere = tmp_path / "nowhere"
        nowhere.mkdir()
        monkeypatch.chdir(nowhere)
        out = runner.invoke(app, ["sessions"]).output
        assert "listing every project" in out
        assert "mine" in out

    def test_site_outside_the_repository_says_where_to_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["site"])
        assert result.exit_code != 0
        assert "repository root" in result.output


class TestMcpSelection:
    @pytest.fixture
    def two_projects(self, isolated_transcripts, projects, monkeypatch):
        a, b = projects
        write(isolated_transcripts, a, "mine", 1000, stuck_session(a))
        write(isolated_transcripts, b, "other", 2000, stuck_session(b, repeats=2))
        monkeypatch.chdir(a)
        return a, b

    def test_every_tool_reports_its_selection(self, two_projects):
        for result in (
            mcp_server.get_session_risk(),
            mcp_server.get_repeated_failures(),
            mcp_server.should_i_restart(),
        ):
            selection = result["selection"]
            assert selection["session_id"] == "mine"
            assert selection["selected_by"] == "project"
            assert selection["transcript_path"].endswith("mine.jsonl")

    def test_risk_uses_the_shipped_model_from_the_project_directory(self, two_projects):
        assert 0.0 <= mcp_server.get_session_risk()["failure_probability"] <= 1.0

    def test_repeated_failures_counts_are_exact(self, two_projects):
        result = mcp_server.get_repeated_failures()
        assert result["recurring_errors"][0]["occurrences"] == 6
        assert result["reissued_calls"][0]["occurrences"] == 6

    def test_list_sessions_is_scoped_to_the_project(self, two_projects):
        scoped = mcp_server.list_sessions()
        assert scoped["scope"] == "project"
        assert [s["session_id"] for s in scoped["sessions"]] == ["mine"]
        everything = mcp_server.list_sessions(all_projects=True)
        assert everything["scope"] == "all"
        assert {s["session_id"] for s in everything["sessions"]} == {"mine", "other"}

    def test_fallback_is_visible_to_the_agent(self, two_projects, tmp_path, monkeypatch):
        nowhere = tmp_path / "nowhere"
        nowhere.mkdir()
        monkeypatch.chdir(nowhere)
        selection = mcp_server.get_repeated_failures()["selection"]
        assert selection["selected_by"] == "fallback"
        assert selection["warnings"]
