"""Published prose must agree with the artifacts it describes.

The page regenerates its tables and charts from `artifacts/`, but README.md
and MODEL_CARD.md are written by hand. A pipeline re-run therefore moves the
figures while the sentences stay where they were, which is exactly how the
calibration section came to quote one pair of Brier scores beside a chart
legend rendering another.

These tests are skipped when the artifacts are absent, so a fresh clone that
has not run the pipeline still has a green suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ("README.md", "MODEL_CARD.md")


def _load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip(f"{name} not built; run the pipeline first")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def gate() -> dict:
    return _load("phase3/gate_cut10.json")


@pytest.fixture(scope="module")
def diagnostics() -> dict:
    return _load("phase4/diagnostics_cut10.json")


@pytest.fixture(scope="module")
def prose() -> str:
    return "\n".join((ROOT / name).read_text() for name in DOCS)


def _quoted(text: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, text))


class TestHeadlineFigures:
    def test_brier_scores_match_the_artifact(self, diagnostics, prose):
        cal = diagnostics["calibration"]
        for key in ("brier_raw", "brier_calibrated"):
            expected = f"{cal[key]:.4f}"
            assert expected in prose, (
                f"{key} is {expected} in the artifact but no document quotes it"
            )

    def test_no_superseded_brier_scores_survive(self, diagnostics, prose):
        cal = diagnostics["calibration"]
        current = {f"{cal[key]:.4f}" for key in ("brier_raw", "brier_calibrated")}
        # Any 4-decimal figure sitting next to the words "Brier score".
        quoted = _quoted(prose, r"Brier score from \*?\*?(\d\.\d{4}) to (\d\.\d{4})")
        for pair in quoted:
            assert set(pair) == current, f"stale Brier pair in prose: {pair}"

    def test_auroc_and_recall_match_the_gate(self, gate, prose):
        best = next(r for r in gate["results"] if r["name"] == gate["gate"]["model"])
        for value in (best["auroc"], best["recall_at_fpr"]):
            assert f"{value:.3f}" in prose, f"{value:.3f} is not quoted anywhere"

    def test_corpus_size_matches_phase_one(self, prose):
        summary = _load("phase1/corpus_summary.json")
        sessions = summary.get("sessions")
        if sessions is None:
            pytest.skip("corpus_summary.json has no session count")
        assert f"{sessions:,}" in prose
