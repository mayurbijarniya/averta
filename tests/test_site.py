"""The results page must be self-contained and must not drop the caveats."""

import json
import re
from html.parser import HTMLParser

import pytest

from averta.site import build


class Balance(HTMLParser):
    VOID = frozenset({"meta", "img", "br", "hr", "input", "link"})

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.mismatches: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.mismatches.append(tag)
        else:
            self.stack.pop()


GATE = {
    "cut_point": 10,
    "diagnostics": {"rows": 4381, "failure_rate": 0.8879},
    "results": [
        {
            "name": "majority",
            "auroc": 0.5,
            "auroc_ci_lower": 0.5,
            "auroc_ci_upper": 0.5,
            "auprc_minority": 0.127,
            "recall_at_fpr": 0.0,
            "brier": 0.25,
        },
        {
            "name": "logistic",
            "auroc": 0.677,
            "auroc_ci_lower": 0.651,
            "auroc_ci_upper": 0.697,
            "auprc_minority": 0.198,
            "recall_at_fpr": 0.191,
            "brier": 0.2267,
        },
    ],
    "gate": {
        "model": "logistic",
        "checks": {"auroc >= 0.65": True, "recall@5% fpr >= 0.25": False},
        "passed": False,
    },
}


@pytest.fixture
def artifacts(tmp_path):
    (tmp_path / "phase3").mkdir()
    (tmp_path / "phase3" / "gate_cut10.json").write_text(json.dumps(GATE))
    return tmp_path


class TestBuild:
    def test_writes_a_page(self, artifacts, tmp_path):
        out = build(artifacts, tmp_path / "index.html")
        assert out.exists()
        assert out.read_text().startswith("<!doctype html>")

    def test_missing_gate_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="averta train"):
            build(tmp_path, tmp_path / "index.html")

    def test_html_is_balanced(self, artifacts, tmp_path):
        parser = Balance()
        parser.feed(build(artifacts, tmp_path / "index.html").read_text())
        assert parser.mismatches == []
        assert parser.stack == []

    def test_no_external_resources(self, artifacts, tmp_path):
        # The page must work offline and from a file:// URL.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "http://" not in text
        assert "https://" not in text or "modelcontextprotocol" not in text
        assert "<script" not in text

    def test_linked_figures_resolve_relative_to_the_page(self, artifacts, tmp_path):
        figures = artifacts / "figures"
        figures.mkdir()
        (figures / "calibration.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        out = tmp_path / "site" / "index.html"
        text = build(artifacts, out).read_text()

        sources = re.findall(r'src="([^"]+)"', text)
        assert sources, "no figure was rendered"
        for source in sources:
            assert not source.startswith("data:")
            assert (out.parent / source).resolve().exists()

    def test_inline_embeds_figures_as_data_uris(self, artifacts, tmp_path):
        figures = artifacts / "figures"
        figures.mkdir()
        (figures / "calibration.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        text = build(artifacts, tmp_path / "index.html", inline=True).read_text()
        assert "data:image/png;base64," in text
        assert 'src="../' not in text

    def test_inline_is_larger_than_linked(self, artifacts, tmp_path):
        figures = artifacts / "figures"
        figures.mkdir()
        (figures / "calibration.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096)

        linked = build(artifacts, tmp_path / "a.html").stat().st_size
        embedded = build(artifacts, tmp_path / "b.html", inline=True).stat().st_size
        assert embedded > linked

    def test_reports_the_gate_failure(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "not met" in text.lower()
        assert "NOT MET" in text

    def test_baselines_are_marked_as_such(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert 'class="base"' in text

    def test_best_model_is_highlighted(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert 'class="pick"' in text

    def test_survives_missing_optional_artifacts(self, artifacts, tmp_path):
        # Only the gate file exists; every other section must be skipped
        # rather than crash or render an empty shell.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert '<section id="results"' in text
        assert '<section id="savings"' not in text
        assert '<section id="transfer"' not in text

    def test_nav_never_links_to_a_skipped_section(self, artifacts, tmp_path):
        # Optional artifacts are absent here, so those sections do not render.
        # Linking to them anyway would leave dead anchors.
        text = build(artifacts, tmp_path / "index.html").read_text()
        for anchor in re.findall(r'<a href="#([a-z-]+)"', text):
            assert f'id="{anchor}"' in text, f"nav links to missing #{anchor}"

    def test_nav_covers_every_rendered_section(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        for anchor in re.findall(r'<section id="([a-z-]+)"', text):
            assert f'href="#{anchor}"' in text, f"#{anchor} missing from nav"

    def test_creates_parent_directories(self, artifacts, tmp_path):
        out = build(artifacts, tmp_path / "nested" / "deep" / "index.html")
        assert out.exists()

    def test_escapes_feature_names(self, artifacts, tmp_path):
        (artifacts / "phase4").mkdir()
        (artifacts / "phase4" / "diagnostics_cut10.json").write_text(
            json.dumps(
                {
                    "importance": [{"feature": "<script>x</script>", "drop": 0.1}],
                    "cost": [
                        {
                            "model": "logistic",
                            "predict_single_ms_p50": 0.18,
                            "predict_single_ms_p95": 0.19,
                            "parameters_bytes": 2150,
                        }
                    ],
                }
            )
        )
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "<script>x</script>" not in text
        assert "&lt;script&gt;" in text


class TestSavingsSection:
    def test_labels_tokens_as_estimated(self, artifacts, tmp_path):
        (artifacts / "phase4").mkdir()
        (artifacts / "phase4" / "savings_cut10.json").write_text(
            json.dumps(
                {
                    "points": [
                        {
                            "threshold": 0.8,
                            "recall": 0.181,
                            "false_positive_rate": 0.047,
                            "savings_rate": 0.081,
                            "successes_terminated": 23,
                        }
                    ]
                }
            )
        )
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "estimated" in text.lower()
        assert "never netted off" in text
