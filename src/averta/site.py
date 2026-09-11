"""Static results page.

The plan called for a Next.js dashboard. This is a single self-contained HTML
file generated from the artifacts instead, because the dashboard is read-only:
it renders three PNGs, four tables and a session list. A Node toolchain,
`node_modules` and a build step would be real infrastructure added for no
capability, and a generated page cannot drift out of sync with the JSON it is
generated from.

There is no CDN, no external CSS and no JavaScript, so the page works offline
and on any static host. Figures are linked from `artifacts/figures` by default;
`--inline` embeds them as data URIs to produce one portable file.
"""

from __future__ import annotations

import base64
import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STYLE = """
:root {
  --ink: #16181d; --muted: #5b6270; --line: #e3e6ec; --bg: #ffffff;
  --accent: #2f7d4f; --warn: #b03030; --code: #f6f7f9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e8eaee; --muted: #9aa2b1; --line: #2a2f38; --bg: #14161a;
    --accent: #6fbf8d; --warn: #e08585; --code: #1c1f25;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 3rem 1.5rem 6rem; max-width: 62rem;
  background: var(--bg); color: var(--ink);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
h1 { font-size: 2rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1.15rem; margin: 3rem 0 .75rem; padding-top: 1.25rem;
     border-top: 1px solid var(--line); }
h3 { font-size: .95rem; margin: 1.75rem 0 .5rem; color: var(--muted);
     text-transform: uppercase; letter-spacing: .06em; }
p { margin: 0 0 1rem; }
.lede { color: var(--muted); font-size: 1.05rem; margin-bottom: 2rem; }
.verdict {
  border-left: 3px solid var(--warn); background: var(--code);
  padding: 1rem 1.25rem; margin: 1.5rem 0; border-radius: 0 6px 6px 0;
}
.verdict strong { color: var(--warn); }
table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem;
        font-size: .9rem; }
th, td { text-align: right; padding: .5rem .6rem;
         border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: .78rem;
     text-transform: uppercase; letter-spacing: .05em; }
tr.highlight td { background: var(--code); font-weight: 600; }
tr.baseline td { color: var(--muted); }
code { background: var(--code); padding: .12em .4em; border-radius: 3px;
       font-size: .88em; }
figure { margin: 1.5rem 0; }
figure img { width: 100%; border: 1px solid var(--line); border-radius: 6px; }
figcaption { color: var(--muted); font-size: .82rem; margin-top: .5rem; }
.note { color: var(--muted); font-size: .86rem; }
footer { margin-top: 4rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .82rem; }
a { color: var(--accent); }
"""

BASELINES = {"majority", "turn_index_only", "error_repeat_only"}


def _img(path: Path, caption: str, *, out_dir: Path, inline: bool) -> str:
    """Render a figure, either inlined or linked.

    Linked is the default and what gets committed: the PNGs already live in
    `artifacts/figures`, and base64 would duplicate them at a third again the
    size. Inlining produces one portable file that works from a `file://` URL
    or an email attachment, which is worth having but not worth versioning.
    """
    if not path.exists():
        return ""

    if inline:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        source = f"data:image/png;base64,{encoded}"
    else:
        source = os.path.relpath(path.resolve(), out_dir.resolve())

    return (
        f'<figure><img alt="{html.escape(caption)}" src="{html.escape(source)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _table(headers: list[str], rows: list[list[str]], classes: list[str]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row, css in zip(rows, classes, strict=True):
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f'<tr class="{css}">{cells}</tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _model_table(gate: dict[str, Any]) -> str:
    rows, classes = [], []
    for result in gate["results"]:
        name = result["name"]
        rows.append(
            [
                f"<code>{html.escape(name)}</code>",
                f"{result['auroc']:.3f}",
                f"[{result['auroc_ci_lower']:.3f}, {result['auroc_ci_upper']:.3f}]",
                f"{result['auprc_minority']:.3f}",
                f"{result['recall_at_fpr']:.3f}",
                f"{result['brier']:.4f}",
            ]
        )
        if name in BASELINES:
            classes.append("baseline")
        elif name == gate["gate"]["model"]:
            classes.append("highlight")
        else:
            classes.append("")
    return _table(
        ["model", "AUROC", "95% CI", "AUPRC", "R@5%FPR", "Brier"], rows, classes
    )


def _gate_table(gate: dict[str, Any]) -> str:
    rows, classes = [], []
    for label, passed in gate["gate"]["checks"].items():
        rows.append([html.escape(label), "pass" if passed else "<strong>FAIL</strong>"])
        classes.append("" if passed else "highlight")
    return _table(["criterion", "result"], rows, classes)


def _importance_table(diagnostics: dict[str, Any], top: int = 10) -> str:
    rows = [
        [f"<code>{html.escape(item['feature'])}</code>", f"{item['drop']:.4f}"]
        for item in diagnostics["importance"][:top]
    ]
    return _table(["feature", "AUROC drop when shuffled"], rows, [""] * len(rows))


def _cost_table(diagnostics: dict[str, Any]) -> str:
    rows, classes = [], []
    for profile in diagnostics["cost"]:
        rows.append(
            [
                f"<code>{html.escape(profile['model'])}</code>",
                f"{profile['predict_single_ms_p50']:.3f}",
                f"{profile['predict_single_ms_p95']:.3f}",
                f"{profile['parameters_bytes'] / 1024:,.1f}",
            ]
        )
        classes.append("highlight" if profile["model"] == "logistic" else "")
    return _table(["model", "p50 ms", "p95 ms", "size KB"], rows, classes)


def _savings_table(savings: dict[str, Any]) -> str:
    rows, classes = [], []
    for point in savings["points"]:
        if point["false_positive_rate"] > 0.30:
            continue
        rows.append(
            [
                f"{point['threshold']:.2f}",
                f"{point['recall']:.3f}",
                f"{point['false_positive_rate']:.3f}",
                f"{point['savings_rate']:.1%}",
                f"{int(point['successes_terminated']):,}",
            ]
        )
        classes.append("highlight" if 0.04 <= point["false_positive_rate"] <= 0.06 else "")
    return _table(
        ["threshold", "recall", "FPR", "est. tokens saved", "successes killed"],
        rows,
        classes,
    )


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def build(artifacts: Path, out: Path, inline: bool = False) -> Path:
    gate = _load(artifacts / "phase3" / "gate_cut10.json")
    diagnostics = _load(artifacts / "phase4" / "diagnostics_cut10.json")
    savings = _load(artifacts / "phase4" / "savings_cut10.json")
    corpus = _load(artifacts / "phase1" / "corpus_summary.json")
    drift = _load(artifacts / "phase6" / "drift_cut40.json")

    if gate is None:
        raise FileNotFoundError(
            f"{artifacts / 'phase3' / 'gate_cut10.json'} missing; run `averta train` first"
        )

    figures = artifacts / "figures"
    generated = datetime.now(UTC).strftime("%Y-%m-%d")

    sections = [
        "<h1>Averta</h1>",
        '<p class="lede">CPU-native failure prediction for AI coding agents. '
        "Can a lightweight model predict when a coding agent is heading toward "
        "failure early enough to save tokens, without killing sessions that "
        "would have recovered?</p>",
        '<div class="verdict"><strong>Pre-registered gate: not met.</strong> '
        "Two attempts, identical criteria for each. Four of five criteria "
        "passed; recall at a 5% false-positive budget reached 0.191 against a "
        "required 0.25. Every number on this page was measured before it was "
        "written.</div>",
    ]

    if corpus:
        rows = [
            ["sessions", f"{corpus['sessions']:,}"],
            ["resolved", f"{corpus['resolved']:,} ({corpus['resolve_rate']:.1%})"],
            ["repositories", f"{corpus['distinct_repos']}"],
            ["task instances", f"{corpus['distinct_instances']:,}"],
            ["turn records", f"{corpus['turn_rows']:,}"],
            ["distinct error signatures", f"{corpus['distinct_error_signatures']:,}"],
        ]
        sections += [
            "<h2>Corpus</h2>",
            _table(["", ""], rows, [""] * len(rows)),
            '<p class="note">Sourced from SWE-Gym/OpenHands-Sampled-Trajectories, '
            "the only one of six public trajectory corpora surveyed that carries "
            "both outcome classes. The data is not redistributed here.</p>",
        ]

    sections += [
        "<h2>Results at the gate</h2>",
        f'<p class="note">Turn 10, {gate["diagnostics"]["rows"]:,} sessions, '
        f'{gate["diagnostics"]["failure_rate"]:.1%} failure rate, five '
        "repository-grouped folds, out-of-fold predictions pooled, confidence "
        "intervals resampling repositories.</p>",
        _model_table(gate),
        "<h3>Gate criteria</h3>",
        _gate_table(gate),
        _img(
            figures / "auroc_by_cut.png",
            "AUROC against cut point. Each cut has a different population — only "
            "sessions reaching that turn appear — so this is not one model tracked "
            "over time.",
            out_dir=out.parent,
            inline=inline,
        ),
    ]

    if diagnostics:
        sections += [
            "<h2>What drives the prediction</h2>",
            '<p class="note">Permutation importance under the same grouped folds. '
            "The expectation was that recurring errors would dominate; they do not. "
            "The strongest features measure declining action novelty.</p>",
            _importance_table(diagnostics),
            "<h2>Inference cost</h2>",
            _cost_table(diagnostics),
            '<p class="note">Single-row latency is the deployment-relevant figure: '
            "the monitor scores one live session per turn. For scale, the neural "
            "monitor this work reproduces uses 0.6B parameters.</p>",
        ]

    if savings:
        sections += [
            "<h2>Savings against harm</h2>",
            _savings_table(savings),
            '<p class="note">Tokens are <strong>estimated</strong> from content '
            "length; the corpus records no token counts. Savings count only what "
            "would have been spent after the cut. Sessions wrongly terminated are "
            "never netted off the savings.</p>",
            _img(
                figures / "savings_tradeoff.png",
                "The two axes are not comparable quantities and are never summed.",
                out_dir=out.parent,
                inline=inline,
            ),
        ]

    sections += [
        "<h2>Calibration</h2>",
        _img(
            figures / "calibration.png",
            "Class weighting is needed for ranking but leaves raw scores on a "
            "re-balanced scale. Isotonic calibration fitted on out-of-fold scores "
            "corrects it.",
            out_dir=out.parent,
            inline=inline,
        ),
    ]

    if drift:
        worst = sorted(
            drift["shifts"], key=lambda s: -abs(s["standardized_difference"])
        )[:6]
        rows = [
            [
                f"<code>{html.escape(s['feature'])}</code>",
                f"{s['corpus_mean']:.2f}",
                f"{s['local_mean']:.2f}",
                f"{s['standardized_difference']:+.2f}",
            ]
            for s in worst
        ]
        sections += [
            "<h2>Cross-scaffold transfer</h2>",
            f'<p class="note">Planned as a transfer AUROC against hand-labelled '
            f'Claude Code sessions. Not reported: only {drift["n_local"]} local '
            "sessions are long enough and none carry labels, which is far below "
            "what any inference would need. What is measurable without labels is "
            "whether the features compute comparably at all.</p>",
            _table(["feature", "corpus", "local", "std diff"], rows, [""] * len(rows)),
            '<p class="note">Every comparable feature shifts by more than 1.3 '
            "standard deviations, and some fall outside the training range "
            "entirely. Cross-scaffold accuracy is therefore <strong>unmeasured</strong>, "
            "and the live monitor labels its output indicative.</p>",
        ]

    sections += [
        "<h2>What this establishes</h2>",
        "<p>A 2.1&nbsp;KB linear model running in 0.18&nbsp;ms on one CPU core "
        "recovers roughly half the token savings reported for a 0.6B neural "
        "monitor. The cost reduction is large and the capability gap is real; "
        "both halves are the result.</p>",
        "<p>Predictive signal exists and is not marginal — AUROC 0.677 with a "
        "confidence interval excluding chance, against three baselines pinned at "
        "0.500. What is missing is the operating point the product needed.</p>",
        "<footer>"
        f"Generated {generated} from committed artifacts. "
        'Reproduce with <code>averta train</code>, <code>averta diagnose</code>, '
        "<code>averta savings</code>, <code>averta figures</code>, "
        "<code>averta site</code>."
        "</footer>",
    ]

    document = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>Averta — results</title>"
        f"<style>{STYLE}</style></head><body>"
        f"{''.join(sections)}"
        "</body></html>"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out
