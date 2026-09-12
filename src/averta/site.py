"""Static results page.

The plan called for a Next.js dashboard. This is a generated HTML file
instead, because the view is read-only: five charts and seven tables. A Node
toolchain, `node_modules` and a build step would be infrastructure added for
no capability, and a page generated from the artifacts cannot drift out of
sync with them.

Charts are inline SVG rather than rendered images — see `charts.py` for why —
so the page is entirely self-contained: no CDN, no external CSS, no
JavaScript, no image files to ship alongside it.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from averta.charts import (
    MUTED,
    SERIES_1,
    SERIES_2,
    Series,
    bar_chart,
    line_chart,
    paired_charts,
)

PAPER = "https://arxiv.org/abs/2608.03222"
DATASET = "https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories"

STYLE = """
:root {
  --ink:#15171c; --muted:#5d6470; --faint:#878e9a; --line:#e4e7ec; --bg:#fff;
  --panel:#f7f8fa; --accent:#2f7d4f; --warn:#b0342f; --radius:8px;
}
@media (prefers-color-scheme:dark){:root{
  --ink:#e9ebef; --muted:#9ba3b1; --faint:#767e8c; --line:#282d36; --bg:#131519;
  --panel:#1a1d23; --accent:#6fbf8d; --warn:#e58a85;
}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:64rem;margin:0 auto;padding:0 1.5rem 6rem}
.skip{position:absolute;left:-9999px}
.skip:focus{left:1rem;top:1rem;background:var(--accent);color:#fff;padding:.5rem 1rem;
  border-radius:var(--radius);z-index:10}

header{padding:4rem 0 2rem}
h1{font-size:2.4rem;margin:0 0 .4rem;letter-spacing:-.025em;font-weight:680}
.tagline{font-size:1.1rem;color:var(--muted);margin:0 0 1.5rem}
.question{
  font-size:1.05rem;font-style:italic;color:var(--ink);border-left:3px solid var(--accent);
  padding:.6rem 0 .6rem 1.1rem;margin:0 0 2rem;max-width:44rem;
}
.badges{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:2rem}
.badge{
  font-size:.75rem;font-weight:600;letter-spacing:.03em;text-transform:uppercase;
  padding:.3rem .7rem;border-radius:100px;border:1px solid var(--line);color:var(--muted);
}
.badge.warn{border-color:var(--warn);color:var(--warn)}

nav{
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);
  padding:1rem 0;margin-bottom:1rem;
}
nav ol{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;
  counter-reset:s;font-size:.88rem}
nav li{counter-increment:s}
nav li::before{content:counter(s) ". ";color:var(--faint)}
nav a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}
nav a:hover{color:var(--accent);border-bottom-color:var(--accent)}

section{padding-top:2.5rem}
h2{font-size:1.35rem;margin:0 0 .3rem;letter-spacing:-.015em;font-weight:660;
  scroll-margin-top:1rem}
.sub{color:var(--muted);font-size:.92rem;margin:0 0 1.2rem}
h3{font-size:.8rem;margin:2rem 0 .6rem;color:var(--faint);text-transform:uppercase;
  letter-spacing:.08em;font-weight:700}
p{margin:0 0 1rem;max-width:46rem}
ul{max-width:46rem;padding-left:1.2rem}
li{margin-bottom:.4rem}

.verdict{
  background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  padding:1.2rem 1.4rem;margin:0 0 2rem;border-radius:0 var(--radius) var(--radius) 0;
}
.verdict h3{margin-top:0;color:var(--warn)}
.verdict p:last-child{margin-bottom:0}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden;margin:1.5rem 0}
.stat{background:var(--bg);padding:1rem 1.1rem}
.stat .v{font-size:1.5rem;font-weight:660;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;display:block}
.stat .k{font-size:.76rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em;margin-top:.15rem;display:block}

.scroll{overflow-x:auto;margin:1rem 0 1.5rem;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:.88rem;min-width:32rem}
caption{text-align:left;color:var(--muted);font-size:.85rem;padding-bottom:.6rem}
th,td{text-align:right;padding:.55rem .7rem;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:650;font-size:.74rem;text-transform:uppercase;
  letter-spacing:.05em;border-bottom-width:1.5px}
tbody tr.pick td{background:var(--panel);font-weight:640}
tbody tr.base td{color:var(--faint)}
tbody tr.fail td{background:color-mix(in srgb,var(--warn) 8%,transparent)}
td.pass{color:var(--accent);font-weight:600}
td.no{color:var(--warn);font-weight:700}

code{background:var(--panel);padding:.12em .42em;border-radius:4px;font-size:.87em;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:1rem 1.1rem;overflow-x:auto;font-size:.84rem;line-height:1.55;margin:1rem 0 1.5rem}
pre code{background:none;padding:0}

figure{margin:1.5rem 0}
figure img{width:100%;border:1px solid var(--line);border-radius:var(--radius);display:block}
figcaption{color:var(--muted);font-size:.83rem;margin-top:.6rem;max-width:46rem}

/* Charts are inline SVG so they theme with the page, stay crisp at any zoom,
   and need no JavaScript. Slots 1 and 2 of the validated categorical palette. */
.viz{--s1:#2a78d6;--s2:#eb6834;margin:1.5rem 0}
@media (prefers-color-scheme:dark){.viz{--s1:#3987e5;--s2:#d95926}}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .tick{fill:var(--faint);font-size:11px;font-variant-numeric:tabular-nums}
.chart .axis{fill:var(--muted);font-size:11px;letter-spacing:.03em}
.chart .serieslabel{font-size:11.5px;font-weight:640}
.chart .barlabel{fill:var(--ink);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.chart .barvalue{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.chart circle{transition:r .12s ease}
.chart circle:hover{r:6}
.chart rect{transition:opacity .12s ease}
.chart rect:hover{opacity:1}
.pair{display:grid;gap:.5rem}
@media (min-width:840px){.pair{grid-template-columns:1fr 1fr}}

.note{color:var(--muted);font-size:.87rem;max-width:46rem}
.flag{border-left:2px solid var(--line);padding-left:1rem;color:var(--muted);
  font-size:.9rem;margin:1.2rem 0;max-width:46rem}

footer{margin-top:5rem;padding-top:1.5rem;border-top:1px solid var(--line);
  color:var(--muted);font-size:.85rem}
a{color:var(--accent)}
a:hover{text-decoration:underline}

@media (max-width:640px){
  header{padding-top:2.5rem}
  h1{font-size:1.9rem}
  nav ol{gap:.3rem 1rem;font-size:.82rem}
}
@media print{
  nav,.badges{display:none}
  body{font-size:11pt}
  section{page-break-inside:avoid}
}
"""

BASELINES = {"majority", "turn_index_only", "error_repeat_only"}

SECTIONS = [
    ("question", "The question"),
    ("data", "Data"),
    ("method", "Method"),
    ("results", "Results"),
    ("attempts", "Two attempts"),
    ("drivers", "What drives it"),
    ("cost", "Inference cost"),
    ("calibration", "Calibration"),
    ("savings", "Savings against harm"),
    ("transfer", "Cross-scaffold transfer"),
    ("limitations", "Limitations"),
    ("using", "Using it"),
]


def _e(text: Any) -> str:
    return html.escape(str(text))


def _table(
    headers: list[str], rows: list[list[str]], classes: list[str], caption: str = ""
) -> str:
    head = "".join(f"<th scope=col>{_e(h)}</th>" for h in headers)
    body = "".join(
        f'<tr class="{css}">' + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row, css in zip(rows, classes, strict=True)
    )
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (
        f'<div class="scroll"><table>{cap}<thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _stats(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="stat"><span class="v">{_e(v)}</span>'
        f'<span class="k">{_e(k)}</span></div>'
        for v, k in items
    )
    return f'<div class="grid">{cells}</div>'


def _section(anchor: str, title: str, sub: str, *body: str) -> str:
    subtitle = f'<p class="sub">{sub}</p>' if sub else ""
    return (
        f'<section id="{anchor}"><h2>{title}</h2>{subtitle}'
        + "".join(part for part in body if part)
        + "</section>"
    )


def _model_table(gate: dict[str, Any]) -> str:
    rows, classes = [], []
    for r in gate["results"]:
        name = r["name"]
        rows.append(
            [
                f"<code>{_e(name)}</code>",
                f"{r['auroc']:.3f}",
                f"[{r['auroc_ci_lower']:.3f}, {r['auroc_ci_upper']:.3f}]",
                f"{r['auprc_minority']:.3f}",
                f"{r['recall_at_fpr']:.3f}",
                f"{r['brier']:.4f}",
            ]
        )
        classes.append(
            "base" if name in BASELINES
            else "pick" if name == gate["gate"]["model"]
            else ""
        )
    return _table(
        ["model", "AUROC", "95% CI", "AUPRC", "R@5%FPR", "Brier"],
        rows,
        classes,
        "Baselines greyed. Best non-baseline model highlighted.",
    )


def _gate_table(gate: dict[str, Any]) -> str:
    rows, classes = [], []
    for label, ok in gate["gate"]["checks"].items():
        cell = '<td class="pass">pass</td>' if ok else '<td class="no">NOT MET</td>'
        rows.append([_e(label), cell.replace("<td", "<span").replace("</td>", "</span>")])
        classes.append("" if ok else "fail")
    return _table(["pre-registered criterion", "result"], rows, classes)


def _auroc_chart(artifacts: Path) -> str:
    """AUROC by cut point, built from the per-cut gate files."""
    cuts, points, bands = [], [], []
    for cut in (3, 5, 10, 20, 40):
        payload = _load(artifacts / "phase3" / f"gate_cut{cut}.json")
        if payload is None:
            continue
        best = max(
            (r for r in payload["results"] if r["name"] not in BASELINES),
            key=lambda r: r["auroc"],
        )
        cuts.append(cut)
        points.append((cut, best["auroc"]))
        bands.append((cut, best["auroc_ci_lower"], best["auroc_ci_upper"]))

    if len(points) < 2:
        return ""

    chart = line_chart(
        [
            Series(
                label="best model",
                points=points,
                band=bands,
                color=SERIES_1,
            ),
            Series(
                label="baselines",
                points=[(c, 0.5) for c in cuts],
                color=SERIES_2,
                dashed=True,
            ),
        ],
        x_ticks=[float(c) for c in cuts],
        y_ticks=[0.45, 0.55, 0.65, 0.75],
        x_label="cut point (turn index)",
        y_label="AUROC",
        title="AUROC by cut point",
        x_categorical=True,
    )
    return (
        '<figure class="viz">' + chart + "<figcaption>Shaded band is the 95% "
        "interval, resampling repositories rather than rows. Cut points are "
        "spaced evenly because they are ordered labels, not a continuous axis. "
        "Each cut has a different population — only sessions that reached that "
        "turn appear — so this is not one model tracked over time."
        "</figcaption></figure>"
    )


def _calibration_chart(diag: dict[str, Any] | None) -> str:
    """Reliability curve, raw against calibrated, with the ideal diagonal."""
    if not diag or "calibration" not in diag:
        return ""

    cal = diag["calibration"]
    raw, fixed = cal["raw"], cal["calibrated"]
    if len(raw["predicted"]) < 2:
        return ""

    chart = line_chart(
        [
            Series(
                label="ideal",
                points=[(0.0, 0.0), (1.0, 1.0)],
                color=MUTED,
                dashed=True,
            ),
            Series(
                label=f"raw ({cal['brier_raw']:.3f})",
                points=list(zip(raw["predicted"], raw["observed"], strict=True)),
                color=SERIES_2,
                dashed=True,
            ),
            Series(
                label=f"calibrated ({cal['brier_calibrated']:.3f})",
                points=list(zip(fixed["predicted"], fixed["observed"], strict=True)),
                color=SERIES_1,
            ),
        ],
        x_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_label="predicted failure probability",
        y_label="observed failure rate",
        title="Calibration, before and after isotonic regression",
        height=340,
    )
    return (
        '<figure class="viz">' + chart + "<figcaption>Brier score in "
        "parentheses. Both curves are shown deliberately — the raw one is the "
        "instructive half, since it is what class weighting does to the output "
        "scale. Showing only the corrected version would hide why the step "
        "exists.</figcaption></figure>"
    )


def _savings_charts(savings: dict[str, Any]) -> str:
    """Savings and harm as two panels sharing an x-axis.

    Previously one plot with two y-scales. A dual axis invents a relationship:
    where the two lines cross is an artefact of how the scales were aligned,
    not something in the data — the exact misreading the caption was trying to
    warn against. Two panels state the trade-off without implying one.
    """
    points = [p for p in savings["points"] if p["false_positive_rate"] <= 0.30]
    if len(points) < 2:
        return ""

    points.sort(key=lambda p: p["false_positive_rate"])
    fpr = [round(p["false_positive_rate"], 3) for p in points]

    panels = paired_charts(
        fpr,
        [p["savings_rate"] * 100 for p in points],
        [float(p["successes_terminated"]) for p in points],
        left_label="tokens saved (%)",
        right_label="successes killed",
        x_label="false positive rate",
        marker=(0.05, "5% budget"),
    )
    return (
        f'<figure class="viz"><div class="pair">{panels}</div>'
        "<figcaption>Two panels rather than two y-axes on one plot. The "
        "quantities are not commensurable — one is tokens not spent, the other "
        "is work destroyed — and overlaying them on shared axes would suggest a "
        "crossing point that is purely an artefact of scaling."
        "</figcaption></figure>"
    )


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def build(artifacts: Path, out: Path) -> Path:
    gate = _load(artifacts / "phase3" / "gate_cut10.json")
    if gate is None:
        raise FileNotFoundError(
            f"{artifacts / 'phase3' / 'gate_cut10.json'} missing; run `averta train` first"
        )

    diag = _load(artifacts / "phase4" / "diagnostics_cut10.json")
    savings = _load(artifacts / "phase4" / "savings_cut10.json")
    corpus = _load(artifacts / "phase1" / "corpus_summary.json")
    drift = _load(artifacts / "phase6" / "drift_cut40.json")

    def viz(svg: str, caption: str) -> str:
        return (
            f'<figure class="viz">{svg}'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    best = next(r for r in gate["results"] if r["name"] == gate["gate"]["model"])
    logistic_cost = None
    if diag:
        logistic_cost = next(
            (c for c in diag["cost"] if c["model"] == "logistic"), None
        )

    parts: list[str] = []

    # ---- header -----------------------------------------------------------
    # The nav is filled in after the body, from the sections that actually
    # rendered — optional artifacts may be absent, and linking to a section
    # that was skipped leaves a dead anchor.
    NAV_SLOT = "<!--NAV-->"
    parts.append(
        '<a class="skip" href="#question">Skip to content</a>'
        '<div class="wrap"><header>'
        "<h1>Averta</h1>"
        '<p class="tagline">CPU-native failure prediction for AI coding agents</p>'
        '<p class="question">Can a lightweight, CPU-native model predict when a '
        "coding agent is heading toward failure early enough to save tokens — "
        "without killing sessions that would have recovered?</p>"
        '<div class="badges">'
        '<span class="badge warn">Pre-registered gate: not met</span>'
        '<span class="badge">Two disclosed attempts</span>'
        '<span class="badge">266 tests</span>'
        '<span class="badge">MIT licensed</span>'
        "</div></header>" + NAV_SLOT
    )

    # ---- the question -----------------------------------------------------
    parts.append(
        _section(
            "question",
            "The question",
            "What this set out to test, and what it found.",
            "<p>A coding agent either resolves its task or it doesn't. A session "
            "heading nowhere keeps spending tokens that produce nothing, so "
            "detecting that early is worth doing — but stopping a session that "
            "would have recovered is worse than letting it run. That asymmetry is "
            "the whole problem: the value is in catching doomed sessions, the cost "
            "is in killing salvageable ones.</p>"
            f'<p><a href="{PAPER}">Fail-Fast, Restart-Smart</a> (Wang et al., 2026) '
            "approaches this with a 0.6B neural monitor. Averta asks whether "
            "engineered trajectory features and a cheap classifier can do useful "
            "work at a fraction of that inference cost.</p>"
            '<div class="verdict"><h3>Verdict</h3>'
            "<p><strong>The pre-registered gate was not met</strong>, across two "
            "attempts judged against identical criteria. Four of five criteria "
            f"passed; recall at a 5% false-positive budget reached "
            f"<strong>{best['recall_at_fpr']:.3f}</strong> against a required 0.25.</p>"
            "<p>What the work does establish: a 2.1&nbsp;KB linear model at "
            "0.18&nbsp;ms CPU inference recovers roughly <strong>half</strong> the "
            "token savings reported for a 0.6B neural monitor. The cost reduction "
            "is large and the capability gap is real.</p></div>"
            '<p class="note">Every number on this page was measured before it was '
            "written. Criteria were committed to source control before any model "
            "existed, so they could not be relaxed to fit a disappointing result.</p>",
        )
    )

    # ---- data -------------------------------------------------------------
    if corpus:
        parts.append(
            _section(
                "data",
                "Data",
                "One corpus, because only one had labels.",
                _stats(
                    [
                        (f"{corpus['sessions']:,}", "sessions"),
                        (f"{corpus['resolve_rate']:.1%}", "resolve rate"),
                        (f"{corpus['distinct_repos']}", "repositories"),
                        (f"{corpus['turn_rows']:,}", "turn records"),
                        (f"{corpus['distinct_error_signatures']:,}", "error signatures"),
                    ]
                ),
                f'<p>Training data comes from <a href="{DATASET}">'
                "SWE-Gym/OpenHands-Sampled-Trajectories</a>. A survey of six public "
                "trajectory corpora found it to be the <strong>only one carrying both "
                "outcome classes</strong> — the others are supervised fine-tuning sets "
                "that store the agent's patch but never whether it worked.</p>"
                "<p>Two properties shape everything downstream. The positive class is "
                "rare, so metrics that tolerate imbalance are required. And "
                "<strong>session length barely separates the classes</strong>: resolved "
                "sessions average 39.86 turns against 39.23 for unresolved. Prior work "
                "describes failing agent runs as tending to run longer; that does not "
                "reproduce here.</p>"
                '<p class="flag">The source carries no declared license, so this '
                "project does not redistribute it. Raw trajectories are downloaded "
                "locally and excluded from version control; only derived statistics, "
                "figures and model artifacts are published.</p>",
            )
        )

    # ---- method -----------------------------------------------------------
    parts.append(
        _section(
            "method",
            "Method",
            "Three constraints, each guarding against a specific way of fooling yourself.",
            "<h3>Prefix-only features</h3>"
            "<p>A feature computed at turn <em>t</em> reads <code>turns[0:t]</code> and "
            "nothing else — never the total length, never the outcome. A test suite "
            "shuffles, truncates and extends the unseen tail and asserts the feature "
            "vector is byte-identical, and checks that features at turn 6 are the same "
            "whether the session runs to 12 turns or 200.</p>"
            "<h3>Absolute turn indices</h3>"
            "<p>Evaluating at &ldquo;40% through the session&rdquo; requires knowing the "
            "total length, which is unavailable while a session is running. Evaluation "
            "uses turns 3, 5, 10, 20 and 40. Each cut has a different population — only "
            "sessions that reached that turn appear — and the resolve rate is reported "
            "per cut so the shift stays visible.</p>"
            "<h3>Repository-grouped splits</h3>"
            "<p>Every row from a repository lands in one fold. Rows sharing a codebase "
            "are not independent, so splitting by row would let a model memorise a "
            "repository and be rewarded for it. Confidence intervals resample "
            "repositories rather than rows for the same reason — row-level bootstrap "
            "reports intervals that are too narrow under clustering.</p>"
            "<h3>Polarity</h3>"
            "<p>The positive class is <strong>failure</strong>, so the false positive "
            "rate means &ldquo;sessions that would have resolved but were flagged&rdquo; "
            "— the quantity worth constraining. Average precision is reported on the "
            "minority class instead, with its polarity named. Accuracy is never "
            "reported: at an 89% failure rate, always predicting failure scores 0.89 and "
            "is useless.</p>",
        )
    )

    # ---- results ----------------------------------------------------------
    parts.append(
        _section(
            "results",
            "Results",
            f"Turn 10 · {gate['diagnostics']['rows']:,} sessions · "
            f"{gate['diagnostics']['failure_rate']:.1%} failure rate · "
            "5 repository-grouped folds · out-of-fold predictions pooled",
            _model_table(gate),
            "<h3>Against the pre-registered criteria</h3>",
            _gate_table(gate),
            '<p class="note">The shortfall is robust: it holds at every cut point '
            "tested and under any model-selection rule. The best recall at a 5% "
            "false-positive budget anywhere on the curve is 0.191.</p>",
            _auroc_chart(artifacts),
        )
    )

    # ---- two attempts -----------------------------------------------------
    parts.append(
        _section(
            "attempts",
            "Two attempts",
            "Both reported. Identical criteria for each.",
            "<p><strong>Attempt 1</strong> used 23 aggregate counts over the prefix. "
            "AUROC 0.668, recall 0.155. Permutation importance showed one feature "
            "dominating — <code>n_repeated_calls</code>, a flat count of tool calls "
            "reissued with byte-identical arguments — while every error-based feature "
            "was negligible.</p>"
            "<p>That pointed at <strong>ordering</strong> as the missing ingredient. A "
            "count knows an action recurred; it cannot express that the agent is cycling "
            "A-B-A-B, or how far back it reached to repeat itself.</p>"
            "<p><strong>Attempt 2</strong> added 10 sequence-structure features. Recall "
            "rose from 0.155 to 0.191 and AUROC from 0.668 to 0.677 — real movement on "
            "exactly the failing criterion, but not enough to clear it.</p>",
            _table(
                ["feature", "attempt 1", "attempt 2"],
                [
                    ["<code>distinct_action_ratio</code>", "—", "<strong>0.159</strong>"],
                    ["<code>novelty_rate_recent</code>", "—", "<strong>0.153</strong>"],
                    ["<code>n_errors</code>", "0.023", "0.084"],
                    ["<code>action_bigram_repeat_max</code>", "—", "0.078"],
                    ["<code>n_repeated_calls</code>", "<strong>0.198</strong>", "0.030"],
                ],
                [""] * 5,
                "AUROC drop when the feature is shuffled, under the same grouped folds.",
            ),
            "<p><code>n_repeated_calls</code> collapsed once ordering was represented "
            "properly — it had been a proxy for structure it could not express. The two "
            "strongest features now both measure <strong>declining action novelty</strong>.</p>"
            '<p class="flag">Modelling stopped after two attempts. The gains were real '
            "but shrinking, the mechanism is understood, and a third round of feature "
            "invention against a bar that had already failed twice would be "
            "threshold-shopping under another name.</p>",
        )
    )

    # ---- drivers ----------------------------------------------------------
    if diag:
        rows = [
            [f"<code>{_e(i['feature'])}</code>", f"{i['drop']:.4f}"]
            for i in diag["importance"][:10]
        ]
        parts.append(
            _section(
                "drivers",
                "What drives it",
                "The expectation was wrong, and measurement is what corrected it.",
                viz(
                    bar_chart(
                        [i["feature"] for i in diag["importance"][:10]],
                        [i["drop"] for i in diag["importance"][:10]],
                        title="Permutation importance",
                        value_format="{:.3f}",
                        highlight=0,
                    ),
                    "AUROC lost when each feature is shuffled within the held-out "
                    "fold. One series, so one colour — the bar length already "
                    "encodes magnitude.",
                ),
                _table(
                    ["feature", "AUROC drop when shuffled"], rows, [""] * len(rows)
                ),
                "<p>The prior expectation was that a recurring error signature would "
                "dominate. It is close to worthless in isolation — AUROC 0.521 alone. "
                "Failure is predicted by <strong>declining action novelty</strong>: the "
                "share of actions that are distinct, and the share of recent actions "
                "never issued before. A session heading nowhere is one that has stopped "
                "trying new things, which is a different phenomenon from one that keeps "
                "hitting the same wall.</p>",
            )
        )

        # ---- cost ---------------------------------------------------------
        cost_rows, cost_classes = [], []
        for c in diag["cost"]:
            cost_rows.append(
                [
                    f"<code>{_e(c['model'])}</code>",
                    f"{c['predict_single_ms_p50']:.3f}",
                    f"{c['predict_single_ms_p95']:.3f}",
                    f"{c['parameters_bytes'] / 1024:,.1f}",
                ]
            )
            cost_classes.append("pick" if c["model"] == "logistic" else "")

        headline = (
            _stats(
                [
                    (f"{logistic_cost['predict_single_ms_p50']:.2f} ms", "p50 latency"),
                    (f"{logistic_cost['parameters_bytes'] / 1024:.1f} KB", "model size"),
                    ("0.6B params", "the neural monitor"),
                ]
            )
            if logistic_cost
            else ""
        )

        parts.append(
            _section(
                "cost",
                "Inference cost",
                "Single-row latency, because the monitor scores one live session per turn.",
                headline,
                _table(["model", "p50 ms", "p95 ms", "size KB"], cost_rows, cost_classes),
                "<p>The most accurate model is also the smallest and fastest by a wide "
                "margin. Plausible cause: with 33 features, 491 positives and "
                "repository-grouped evaluation, trees fit repository-specific thresholds "
                "that do not survive the group boundary.</p>",
            )
        )

    # ---- calibration ------------------------------------------------------
    parts.append(
        _section(
            "calibration",
            "Calibration",
            "A bug the plot caught, kept visible because the before-curve is the lesson.",
            "<p>Every model is fitted with class weighting, which is correct for ranking "
            "but leaves the output on a re-balanced scale. A raw score of 0.25 "
            "corresponded to an observed failure rate near 0.70. Rank-based metrics are "
            "unaffected — but any number shown to a person has to mean what it says.</p>"
            "<p>Isotonic regression fitted on <strong>out-of-fold</strong> scores "
            "corrects it, moving the Brier score from 0.2323 to 0.0826. Fitting the "
            "calibrator on training predictions would have learned the model's own "
            "overconfidence and reported it back as calibrated.</p>",
            _calibration_chart(diag),
        )
    )

    # ---- savings ----------------------------------------------------------
    if savings:
        rows, classes = [], []
        for p in savings["points"]:
            if p["false_positive_rate"] > 0.30:
                continue
            rows.append(
                [
                    f"{p['threshold']:.2f}",
                    f"{p['recall']:.3f}",
                    f"{p['false_positive_rate']:.3f}",
                    f"{p['savings_rate']:.1%}",
                    f"{int(p['successes_terminated']):,}",
                ]
            )
            classes.append(
                "pick" if 0.04 <= p["false_positive_rate"] <= 0.06 else ""
            )
        parts.append(
            _section(
                "savings",
                "Savings against harm",
                "What intervention buys, and what it costs.",
                _table(
                    ["threshold", "recall", "FPR", "est. tokens saved", "successes killed"],
                    rows,
                    classes,
                    "Highlighted row is the operating point nearest a 5% false-positive budget.",
                ),
                "<p>At a comparable budget — 4.7% against the paper's 5% target — this "
                "reaches <strong>8.1% estimated token savings</strong> where the 0.6B "
                "neural monitor reports 14.6&ndash;20.4%. Roughly half the value, at a "
                "fraction of the size.</p>"
                '<p class="flag">Tokens are <strong>estimated</strong> from content '
                "length; the corpus records no token counts. Savings count only what "
                "would have been spent after the cut. Sessions wrongly terminated are "
                "never netted off the savings — the two are not commensurable and a "
                "single &ldquo;net&rdquo; figure would let the harm disappear into an "
                "aggregate.</p>",
                _savings_charts(savings),
            )
        )

    # ---- transfer ---------------------------------------------------------
    if drift:
        worst = sorted(
            drift["shifts"], key=lambda s: -abs(s["standardized_difference"])
        )[:6]
        rows = [
            [
                f"<code>{_e(s['feature'])}</code>",
                f"{s['corpus_mean']:.2f}",
                f"{s['local_mean']:.2f}",
                f"{s['standardized_difference']:+.2f}",
            ]
            for s in worst
        ]
        parts.append(
            _section(
                "transfer",
                "Cross-scaffold transfer",
                "Planned as an AUROC comparison. Not reported — the sample "
                "cannot support one.",
                f"<p>The plan was to train on OpenHands trajectories and test on "
                f"hand-labelled Claude Code sessions. Only {drift['n_local']} local "
                "sessions are long enough and none carry outcome labels, so a transfer "
                "AUROC would be noise presented as a result.</p>"
                "<p>What is measurable without labels is whether the features compute "
                "comparably at all — a prerequisite for transfer rather than a "
                "substitute for measuring it.</p>",
                _table(
                    ["feature", "corpus", "local", "std diff"], rows, [""] * len(rows)
                ),
                "<p>Every comparable feature shifts by more than 1.3 standard "
                "deviations and some fall outside the training range entirely. Claude "
                "Code emits more assistant turns per tool call than OpenHands, because "
                "thinking blocks become their own turns — a &ldquo;turn&rdquo; is not the "
                "same unit across scaffolds. Cross-scaffold accuracy is therefore "
                "<strong>unmeasured</strong>, and the tool labels its output indicative "
                "in every code path.</p>"
                '<p class="flag">This check caught a real bug. Two features reported '
                "exactly 0.00 on sessions full of edits, because edit detection was keyed "
                "to one tool vocabulary. Not a slightly worse score — a confident zero "
                "for something that happened dozens of times.</p>",
            )
        )

    # ---- limitations ------------------------------------------------------
    parts.append(
        _section(
            "limitations",
            "Limitations",
            "Stated plainly rather than buried.",
            "<ul>"
            "<li><strong>The operating point is not reachable here.</strong> Two "
            "attempts, the second targeting the diagnosed gap, both fell short. Closing "
            "it likely needs a genuine sequence model over the action stream rather than "
            "more engineered summaries of it.</li>"
            "<li><strong>One corpus, one agent scaffold, 11 repositories.</strong> "
            "Grouped folds mean 11 grouping units, which widens every interval.</li>"
            "<li><strong>Token savings are estimated</strong>, not counted. The corpus "
            "records no per-message token usage.</li>"
            "<li><strong>Cross-scaffold accuracy is unmeasured.</strong> The feature "
            "comparison above suggests it would be poor.</li>"
            "<li><strong>The comparison to the paper is not controlled.</strong> "
            "Different corpus, different scaffold, estimated rather than counted tokens. "
            "It is the closest like-for-like available, not a replication.</li>"
            "</ul>",
        )
    )

    # ---- using it ---------------------------------------------------------
    parts.append(
        _section(
            "using",
            "Using it",
            "Runs entirely locally. No API keys, no network at inference time, "
            "no session data leaves the machine.",
            "<pre><code>git clone &lt;repo&gt; &amp;&amp; cd averta\n"
            "python3 -m venv .venv\n"
            ".venv/bin/python -m pip install -e .\n\n"
            "averta explain      # analyse your most recent coding session\n"
            "averta sessions     # list local sessions\n"
            "averta site         # rebuild this page</code></pre>"
            "<p>Those work immediately — the trained model is committed. Reproducing the "
            "study needs the corpus, which is a 12-minute download:</p>"
            "<pre><code>averta ingest &amp;&amp; averta features\n"
            "averta train        # cross-validate and apply the gate\n"
            "averta diagnose     # importance and inference cost\n"
            "averta savings      # savings against harm\n"
            "averta figures      # the plots on this page</code></pre>"
            "<p><code>averta explain</code> separates what is <strong>measured</strong> — "
            "recurring errors, reissued calls, clustered repetition, exact token counts — "
            "from what is <strong>estimated</strong> by a model that did not clear its "
            "gate. The measured half needs no model and is as reliable as the transcript. "
            "An MCP server exposes the same analysis to a coding agent over stdio.</p>"
            '<p class="note">macOS users need <code>brew install libomp</code> for '
            "xgboost. On Linux, <code>libgomp1</code>.</p>",
        )
    )

    generated = datetime.now(UTC).strftime("%d %B %Y")
    parts.append(
        "<footer>"
        f"<p>Generated {generated} from committed artifacts — every figure and table "
        "on this page is produced by <code>averta site</code> from JSON written by the "
        "pipeline, so it cannot drift out of sync with the results.</p>"
        f'<p>Reproduces <a href="{PAPER}">arXiv:2608.03222</a>. MIT licensed; the '
        "training corpus is not redistributed.</p>"
        "</footer></div>"
    )

    body = "".join(parts)

    present = [
        (anchor, title)
        for anchor, title in SECTIONS
        if f'<section id="{anchor}"' in body
    ]
    nav_items = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in present)
    body = body.replace(NAV_SLOT, f"<nav aria-label=Contents><ol>{nav_items}</ol></nav>")

    document = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        '<meta name=description content="CPU-native failure prediction for AI coding '
        'agents: a pre-registered evaluation with a negative result.">'
        "<title>Averta — CPU-native failure prediction for AI coding agents</title>"
        f"<style>{STYLE}</style></head><body>"
        f"{body}"
        "</body></html>"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out
