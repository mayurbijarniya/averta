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


def _table_rows(text: str) -> list[list[str]]:
    """Every Markdown table row as its cells, bold and code markers stripped."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([cell.strip().strip("*`") for cell in line.strip("|").split("|")])
    return rows


MODEL_LABELS = {
    "majority": "majority",
    "turn index only": "turn_index_only",
    "logistic": "logistic",
    "random forest": "random_forest",
    "hist gradient boosting": "hist_gradient_boosting",
    "xgboost": "xgboost",
}


def _size(n_bytes: int) -> str:
    kb = n_bytes / 1024
    if kb < 10:
        return f"{kb:.1f} KB"
    if kb < 1024:
        return f"{kb:.0f} KB"
    return f"{kb / 1024:.1f} MB"


def _latency(ms: float) -> str:
    return f"{ms:.2f} ms" if ms < 1 else f"{ms:.1f} ms"


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

    def test_cost_figures_match_the_diagnostics(self, gate, diagnostics, prose):
        cost = next(
            c for c in diagnostics["cost"] if c["model"] == gate["gate"]["model"]
        )
        size = f"{cost['parameters_bytes'] / 1024:.1f} KB"
        latency = f"{cost['predict_single_ms_p50']:.2f} ms"
        for value in (size, latency):
            plain = value.replace(" ", "")
            assert value in prose or plain in prose, f"{value} is not quoted anywhere"

    def test_mean_turn_counts_match_phase_one(self, prose):
        summary = _load("phase1/corpus_summary.json")
        for row in summary.get("by_outcome", []):
            expected = f"{row['mean_turns']:.2f}"
            assert expected in prose, (
                f"mean turns for resolved={row['resolved']} is {expected} "
                "in the artifact but no document quotes it"
            )

    def test_savings_figure_matches_the_simulation(self, prose):
        savings = _load("phase4/savings_cut10.json")
        eligible = [p for p in savings["points"] if p["false_positive_rate"] <= 0.05]
        if not eligible:
            pytest.skip("no operating point within the false-positive budget")
        point = max(eligible, key=lambda p: p["savings_rate"])
        assert f"{point['savings_rate'] * 100:.1f}%" in prose

    def test_drift_table_matches_the_artifact(self, prose):
        # The transfer table is hand-copied from the drift artifact, which a
        # reader fix or new local sessions can move without touching prose.
        drift = _load("phase6/drift_cut40.json")
        shifts = {s["feature"]: s for s in drift["shifts"]}
        rows = re.findall(
            r"^\| `(\w+)` \| (\d+\.\d{2}) \| (\d+\.\d{2}) \| ([+−-]\d+\.\d{2})",
            prose,
            flags=re.MULTILINE,
        )
        assert rows, "no drift table found in the documents"
        for feature, corpus, local, std_diff in rows:
            shift = shifts[feature]
            assert corpus == f"{shift['corpus_mean']:.2f}", feature
            assert local == f"{shift['local_mean']:.2f}", feature
            assert std_diff.replace("−", "-") == f"{shift['standardized_difference']:+.2f}", feature

    def test_drift_sample_sizes_match_the_artifact(self, prose):
        drift = _load("phase6/drift_cut40.json")
        quoted = _quoted(prose, r"local \(n=(\d+)\)")
        assert quoted == {str(drift["n_local"])}, f"local sample sizes quoted: {quoted}"
        assert f"corpus (n={drift['n_corpus']:,})" in prose

    def test_model_table_matches_the_gate(self, gate, diagnostics, prose):
        results = {r["name"]: r for r in gate["results"]}
        cost = {c["model"]: c for c in diagnostics["cost"]}
        rows = [r for r in _table_rows(prose) if r[0] in MODEL_LABELS and len(r) >= 7]
        assert {r[0] for r in rows} >= {"logistic", "random forest", "xgboost"}, rows
        for label, auroc, ci, auprc, recall, latency, size, *_ in rows:
            name = MODEL_LABELS[label]
            result = results[name]
            assert auroc == f"{result['auroc']:.3f}", name
            assert ci == f"[{result['auroc_ci_lower']:.3f}, {result['auroc_ci_upper']:.3f}]", name
            assert auprc == f"{result['auprc_minority']:.3f}", name
            assert recall == f"{result['recall_at_fpr']:.3f}", name
            if name in cost:
                assert latency == _latency(cost[name]["predict_single_ms_p50"]), name
                assert size == _size(cost[name]["parameters_bytes"]), name

    def test_importance_tables_match_the_diagnostics(self, diagnostics, prose):
        # The attempt-2 column, and the model card's single column, are the
        # current model. Attempt 1 is a superseded run and is not checked.
        drops = {i["feature"]: i["drop"] for i in diagnostics["importance"]}
        checked = 0
        for row in _table_rows(prose):
            # Drift rows also start with a feature name but carry more columns.
            if row[0] in drops and len(row) in (2, 3) and re.fullmatch(r"\d\.\d{3}", row[-1]):
                assert row[-1] == f"{drops[row[0]]:.3f}", row[0]
                checked += 1
        assert checked >= 10, f"expected both importance tables, checked {checked} rows"

    def test_savings_table_matches_the_simulation(self, prose):
        savings = _load("phase4/savings_cut10.json")
        points = {f"{p['threshold']:.2f}": p for p in savings["points"]}
        rows = [r for r in _table_rows(prose) if r[0] in points and len(r) == 6]
        assert rows, "no savings table found in the documents"
        for threshold, recall, fpr, tokens, share, killed in rows:
            point = points[threshold]
            assert recall == f"{point['recall']:.3f}", threshold
            assert fpr == f"{point['false_positive_rate']:.3f}", threshold
            assert tokens == f"{point['estimated_tokens_saved']:,}", threshold
            assert share == f"{point['savings_rate'] * 100:.1f}%", threshold
            assert killed == str(point["successes_terminated"]), threshold

    def test_corpus_size_matches_phase_one(self, prose):
        summary = _load("phase1/corpus_summary.json")
        sessions = summary.get("sessions")
        if sessions is None:
            pytest.skip("corpus_summary.json has no session count")
        assert f"{sessions:,}" in prose
