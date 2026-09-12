"""The results page must be self-contained and must not drop the caveats."""

import json
import re
from html.parser import HTMLParser
from xml.etree import ElementTree

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
    """The minimum the page needs: one gate file, nothing optional."""
    (tmp_path / "phase3").mkdir()
    (tmp_path / "phase3" / "gate_cut10.json").write_text(json.dumps(GATE))
    return tmp_path


@pytest.fixture
def full_artifacts(artifacts):
    """Enough for every chart to render, the AUROC curve needs several cuts."""
    for cut in (3, 5, 20, 40):
        payload = json.loads(json.dumps(GATE))
        payload["cut_point"] = cut
        (artifacts / "phase3" / f"gate_cut{cut}.json").write_text(json.dumps(payload))

    (artifacts / "phase4").mkdir()
    (artifacts / "phase4" / "diagnostics_cut10.json").write_text(
        json.dumps(
            {
                "importance": [
                    {"feature": "distinct_action_ratio", "drop": 0.159},
                    {"feature": "novelty_rate_recent", "drop": 0.153},
                ],
                "cost": [
                    {
                        "model": "logistic",
                        "predict_single_ms_p50": 0.18,
                        "predict_single_ms_p95": 0.19,
                        "parameters_bytes": 2150,
                    }
                ],
                "calibration": {
                    "raw": {
                        "predicted": [0.2, 0.5, 0.8],
                        "observed": [0.7, 0.9, 0.95],
                        "counts": [10, 20, 30],
                    },
                    "calibrated": {
                        "predicted": [0.6, 0.8, 0.95],
                        "observed": [0.6, 0.8, 0.94],
                        "counts": [10, 20, 30],
                    },
                    "brier_raw": 0.232,
                    "brier_calibrated": 0.083,
                },
            }
        )
    )
    (artifacts / "phase4" / "savings_cut10.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "threshold": t,
                        "recall": r,
                        "false_positive_rate": f,
                        "savings_rate": s,
                        "successes_terminated": k,
                    }
                    for t, r, f, s, k in [
                        (0.90, 0.05, 0.010, 0.024, 5),
                        (0.80, 0.18, 0.047, 0.081, 23),
                        (0.65, 0.28, 0.096, 0.159, 47),
                        (0.55, 0.50, 0.246, 0.336, 121),
                    ]
                ]
            }
        )
    )
    return artifacts


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
        # Outbound *links* are fine and wanted; what must not exist is a
        # fetched resource, which would break the page offline. The previous
        # version of this test was vacuous and asserted nothing.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "<script" not in text
        assert "<link" not in text
        for tag in re.findall(r"<(?:img|iframe|source|embed)\b[^>]*>", text):
            assert "http" not in tag, f"fetches an external resource: {tag}"
        assert not re.search(r'@import|url\(\s*["\']?http', text)

    def test_no_em_dashes(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "—" not in text

    def test_small_screens_get_a_drawer_not_a_pill_strip(self, artifacts, tmp_path):
        # The nav carries a dozen entries. As a horizontal strip it overflowed
        # every phone width, so below the breakpoint it becomes an off-canvas
        # drawer opened by :target, which needs no JavaScript.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert 'href="#menu"' in text
        assert 'id="menu"' in text
        assert "aside:target" in text
        # The trigger and the scrim must both be inside a max-width block,
        # otherwise they leak onto desktop.
        small = "\n".join(re.findall(r"@media\s*\(max-width:[^{]*\{.*?\}\s*\}", text, re.S))
        assert ".burger" in small
        assert "translateX" in small
        # The scrim must be a sibling of the drawer, not a child. A child
        # would take the open drawer's transform as its containing block and
        # shrink to the drawer's own width.
        assert "aside:target ~ .scrim" in small
        assert re.search(r"</aside>\s*<a class=\"scrim\"", text)

    def test_every_outbound_link_opens_in_a_new_tab(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        for tag in re.findall(r"<a\b[^>]*href=\"https?://[^>]*>", text):
            assert 'target="_blank"' in tag, tag
            assert "noopener" in tag, tag

    def test_links_to_the_source(self, artifacts, tmp_path):
        # A reader who arrives at the results must be able to reach the code.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "github.com" in text
        assert "MODEL_CARD" in text

    def test_outbound_links_open_in_a_new_tab(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        external = re.findall(r'<a\b[^>]*href="https?://[^"]+"[^>]*>', text)
        assert external
        for tag in external:
            assert 'target="_blank"' in tag, tag
            # target=_blank without noopener hands window.opener to the
            # destination, which is a real security footgun.
            assert 'rel="noopener noreferrer"' in tag, tag

    def test_in_page_anchors_do_not_open_a_new_tab(self, artifacts, tmp_path):
        text = build(artifacts, tmp_path / "index.html").read_text()
        for tag in re.findall(r'<a\b[^>]*href="#[^"]*"[^>]*>', text):
            assert "target=" not in tag, tag

    def test_no_duplicated_call_to_action(self, artifacts, tmp_path):
        # Two identical "View source" controls was noise; the sidebar carries
        # the repository handle instead.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert text.count(">View source") == 1

    def test_responsive_breakpoints_are_defined(self, artifacts, tmp_path):
        # Phone, tablet and laptop are handled explicitly rather than left to
        # a single grid that degrades.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "max-width:380px" in text
        assert "max-width:700px" in text
        assert "max-width:959px" in text
        assert "min-width:960px" in text
        assert "width=device-width" in text

    def test_no_hard_coded_test_count(self, artifacts, tmp_path):
        # A count baked into the page goes stale the moment a test is added,
        # and did: the hero claimed 269 while the suite was at 272.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert not re.search(r"\d{2,4}\s+tests", text)

    def test_charts_are_inline_svg(self, full_artifacts, tmp_path):
        # No raster figures: the page must carry its own charts so it themes
        # with the surface and stays crisp at any zoom.
        text = build(full_artifacts, tmp_path / "index.html").read_text()
        assert "<svg" in text
        assert "<img" not in text
        assert "data:image" not in text

    def test_every_chart_is_well_formed_xml(self, full_artifacts, tmp_path):
        text = build(full_artifacts, tmp_path / "index.html").read_text()
        charts = re.findall(r"<svg\b.*?</svg>", text, re.S)
        assert charts
        for chart in charts:
            ElementTree.fromstring(chart)

    def test_charts_theme_through_css_variables(self, full_artifacts, tmp_path):
        # Hard-coded fills would not follow prefers-color-scheme.
        text = build(full_artifacts, tmp_path / "index.html").read_text()
        charts = "".join(re.findall(r"<svg\b.*?</svg>", text, re.S))
        assert "var(--s1)" in charts or "var(--muted)" in charts

    def test_charts_have_accessible_labels(self, full_artifacts, tmp_path):
        text = build(full_artifacts, tmp_path / "index.html").read_text()
        charts = [s for s in re.findall(r"<svg\b[^>]*>", text) if 'class="chart"' in s]
        assert charts
        for chart in charts:
            assert 'role="img"' in chart
            assert "aria-label=" in chart

    def test_icons_are_hidden_from_assistive_tech(self, full_artifacts, tmp_path):
        # Icons are decorative, every one sits beside a text label, so
        # announcing them would just add noise for a screen reader.
        text = build(full_artifacts, tmp_path / "index.html").read_text()
        icons = [s for s in re.findall(r"<svg\b[^>]*>", text) if "lucide" in s]
        assert icons
        for glyph in icons:
            assert 'aria-hidden="true"' in glyph
            assert 'role="img"' not in glyph

    def test_icons_inherit_colour(self, full_artifacts, tmp_path):
        # A hard-coded stroke would not follow the theme or the parent's state.
        text = build(full_artifacts, tmp_path / "index.html").read_text()
        for glyph in re.findall(r'<svg[^>]*class="lucide[^>]*>', text):
            assert 'stroke="currentColor"' in glyph

    def test_gate_result_is_not_colour_alone(self, artifacts, tmp_path):
        # The verdict must survive a colour-vision deficiency: icon + word.
        text = build(artifacts, tmp_path / "index.html").read_text()
        assert "NOT MET" in text
        assert ">pass<" in text

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
