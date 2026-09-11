import json

import numpy as np
import pytest

from averta.adapters.claude_code import discover_transcripts, read_transcript
from averta.features import FEATURE_NAMES
from averta.monitor import (
    MIN_TURNS_TO_SCORE,
    Calibrator,
    RiskReport,
    Scorer,
    fit_and_save,
)
from tests.factories import session

CUTS = (3, 5, 10, 20, 40)


def write_transcript(tmp_path, records):
    project = tmp_path / "-some-project"
    project.mkdir()
    path = project / "abc123.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


def assistant(text="ok", tool=None, tool_input=None, usage=None):
    content = [{"type": "text", "text": text}]
    if tool:
        content.append({"type": "tool_use", "name": tool, "input": tool_input or {}})
    message = {"role": "assistant", "content": content}
    if usage:
        message["usage"] = usage
    return {"type": "assistant", "message": message, "cwd": "/home/me/proj"}


def tool_result(body="done", is_error=False):
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": body, "is_error": is_error}
            ],
        },
        "cwd": "/home/me/proj",
    }


class TestClaudeCodeAdapter:
    def test_parses_assistant_and_tool_turns(self, tmp_path):
        path = write_transcript(
            tmp_path,
            [
                assistant("thinking", tool="Bash", tool_input={"command": "ls"}),
                tool_result("file listing"),
            ],
        )
        transcript = read_transcript(path)
        assert len(transcript) == 2
        assert transcript.turns[0].role == "assistant"
        assert transcript.turns[0].tool_name == "Bash"
        assert transcript.turns[1].role == "tool"

    def test_uses_explicit_error_flag(self, tmp_path):
        path = write_transcript(tmp_path, [tool_result("all good", is_error=True)])
        transcript = read_transcript(path)
        assert transcript.turns[0].is_error

    def test_clean_result_is_not_an_error(self, tmp_path):
        path = write_transcript(tmp_path, [tool_result("ran 12 tests, all passed")])
        assert not read_transcript(path).turns[0].is_error

    def test_accumulates_token_usage(self, tmp_path):
        path = write_transcript(
            tmp_path,
            [assistant(usage={"input_tokens": 100, "output_tokens": 20})],
        )
        transcript = read_transcript(path)
        assert transcript.input_tokens == 100
        assert transcript.output_tokens == 20
        assert transcript.total_tokens == 120

    def test_skips_editor_bookkeeping_records(self, tmp_path):
        path = write_transcript(
            tmp_path,
            [
                {"type": "file-history-snapshot", "snapshot": {}},
                {"type": "mode", "mode": "default"},
                {"type": "ai-title", "aiTitle": "x"},
                assistant(),
            ],
        )
        assert len(read_transcript(path)) == 1

    def test_counts_unparseable_lines(self, tmp_path):
        project = tmp_path / "-proj"
        project.mkdir()
        path = project / "s.jsonl"
        path.write_text('{"type":"assistant"\n' + json.dumps(assistant()))
        transcript = read_transcript(path)
        assert transcript.malformed_lines == 1
        assert len(transcript) == 1

    def test_repo_derived_from_cwd(self, tmp_path):
        path = write_transcript(tmp_path, [assistant()])
        assert read_transcript(path).repo == "proj"

    def test_discovery_returns_empty_for_missing_root(self, tmp_path):
        assert discover_transcripts(tmp_path / "nope") == []


@pytest.fixture
def scorer(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((300, len(FEATURE_NAMES)))
    y = (rng.random(300) < 0.4).astype(int)
    return fit_and_save(X, y, cut_points=CUTS, path=tmp_path / "m.pkl")


class TestCalibrator:
    def test_uncalibrated_passes_scores_through(self):
        raw = np.array([0.1, 0.5, 0.9])
        assert np.allclose(Calibrator().apply(raw), raw)

    def test_corrects_systematic_underprediction(self):
        from averta.metrics import brier_score

        rng = np.random.default_rng(0)
        truth = (rng.random(2000) < 0.9).astype(int)
        # Class weighting produces scores centred far below the true rate.
        raw = np.clip(truth * 0.25 + 0.2 + rng.normal(0, 0.05, 2000), 0, 1)

        calibrated = Calibrator().fit(raw, truth).apply(raw)
        assert brier_score(truth, calibrated) < brier_score(truth, raw)

    def test_output_stays_in_range(self):
        rng = np.random.default_rng(1)
        raw = rng.random(500)
        truth = (rng.random(500) < 0.5).astype(int)
        out = Calibrator().fit(raw, truth).apply(np.array([-1.0, 0.5, 2.0]))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_preserves_ranking(self):
        rng = np.random.default_rng(2)
        raw = rng.random(300)
        truth = (rng.random(300) < raw).astype(int)
        calibrated = Calibrator().fit(raw, truth).apply(raw)
        # Isotonic is monotone, so order can flatten but never invert.
        order = np.argsort(raw)
        assert np.all(np.diff(calibrated[order]) >= -1e-9)


def report_with(probability: float | None, base_rate: float | None = 0.88) -> RiskReport:
    return RiskReport(
        session_id="s",
        turns=20,
        failure_probability=probability,
        drivers=(),
        features={},
        base_rate=base_rate,
    )


class TestRiskContext:
    def test_at_the_base_rate_reports_no_signal(self):
        # 89% failure looks alarming until you know 88% of sessions fail.
        assert "no clear signal" in report_with(0.89).verdict

    def test_above_base_rate_reports_worse(self):
        report = report_with(0.97)
        assert report.verdict == "worse than typical"
        assert report.lift > 1.0

    def test_below_base_rate_reports_better(self):
        assert report_with(0.60).verdict == "better than typical"

    def test_render_includes_the_base_rate(self):
        assert "base rate" in report_with(0.9).render()

    def test_lift_is_none_without_a_base_rate(self):
        assert report_with(0.9, base_rate=None).lift is None

    def test_unscored_session_has_no_verdict(self):
        assert report_with(None).verdict == "not scored"


class TestScorer:
    def test_refuses_to_score_very_short_sessions(self, scorer):
        report = scorer.score("s", session(4)[:MIN_TURNS_TO_SCORE - 1])
        assert report.failure_probability is None
        assert "chance" in report.note

    def test_scores_once_long_enough(self, scorer):
        report = scorer.score("s", session(12))
        assert 0.0 <= report.failure_probability <= 1.0

    def test_truncates_to_the_largest_trained_cut(self, scorer):
        report = scorer.score("s", session(500))
        assert report.scored_at_turn == 40
        assert report.turns == 500
        assert "reflects the first 40 turns" in report.note

    def test_does_not_truncate_within_range(self, scorer):
        report = scorer.score("s", session(20))
        assert report.scored_at_turn == 20

    def test_long_and_truncated_sessions_agree(self, scorer):
        # Truncation must make the score independent of the unseen tail.
        turns = session(400)
        assert scorer.score("a", turns).failure_probability == pytest.approx(
            scorer.score("b", turns[:40]).failure_probability
        )

    def test_transfer_warning_is_always_present(self, scorer):
        assert "unmeasured" in scorer.score("s", session(12)).render()

    def test_reports_driver_features(self, scorer):
        report = scorer.score("s", session(12), top=3)
        assert len(report.drivers) == 3
        assert all(driver.feature in FEATURE_NAMES for driver in report.drivers)

    def test_round_trips_through_disk(self, scorer, tmp_path):
        reloaded = Scorer.load(tmp_path / "m.pkl")
        assert reloaded.feature_names == scorer.feature_names
        assert reloaded.cut_points == CUTS

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Scorer.load(tmp_path / "absent.pkl")
