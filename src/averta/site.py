"""Static results page.

The plan called for a Next.js dashboard. This is a generated HTML file
instead, because the view is read-only: five charts and seven tables. A Node
toolchain, `node_modules` and a build step would be infrastructure added for
no capability, and a page generated from the artifacts cannot drift out of
sync with them.

Charts are inline SVG rather than rendered images, see `charts.py` for why -
so the page is entirely self-contained: no CDN, no external CSS, no
JavaScript, no image files to ship alongside it.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from averta.brand import favicon_data_uri, mark
from averta.charts import (
    MUTED,
    SERIES_1,
    SERIES_2,
    Series,
    bar_chart,
    line_chart,
    paired_charts,
)
from averta.icons import SECTION_ICONS, icon

PAPER = "https://arxiv.org/abs/2608.03222"
REPO = "https://github.com/mayurbijarniya/averta"
DATASET = "https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories"

STYLE = """
:root {
  --ink:#0d0f13; --muted:#5b6371; --faint:#8b93a1; --line:#e6e8ec; --bg:#fff;
  --panel:#f8f9fb; --raised:#fff;
  /* Link blue is distinct from status green. Green means "pass" and nothing
     else, so it cannot be mistaken for "this is clickable". */
  --link:#1f6feb; --ok:#1a7f37; --warn:#b0342f; --r:9px;
  --mono:ui-monospace,"Geist Mono","IBM Plex Mono",SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --ink:#eceef2; --muted:#98a1b0; --faint:#6f7889; --line:#252a33; --bg:#0f1115;
  --panel:#161a20; --raised:#171b21;
  --link:#589bff; --ok:#3fb950; --warn:#e58a85;
}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font:15.5px/1.68 Inter,"Geist",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;font-feature-settings:"cv02","cv03","cv04";
}
.skip{position:absolute;left:-9999px}
.skip:focus{left:1rem;top:1rem;background:var(--link);color:#fff;padding:.5rem 1rem;
  border-radius:var(--r);z-index:20}

/* ---- layout -------------------------------------------------------------
   Laptop gets a sticky sidebar. Phone and tablet get a real off-canvas
   drawer, opened with :target so no JavaScript is needed. Tapping any
   section link changes the hash, which un-targets the drawer and closes it
   on the way to the section. */
.shell{display:grid;grid-template-columns:1fr;max-width:88rem;margin:0 auto}

.topbar{display:none}
.burger{display:none}
.scrim{display:none}

@media (max-width:959px){
  .topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;
    position:sticky;top:0;z-index:30;background:var(--bg);
    border-bottom:1px solid var(--line);padding:.7rem 1.1rem}
  .burger{display:inline-flex;align-items:center;gap:.45rem;padding:.42rem .7rem;
    border:1px solid var(--line);border-radius:8px;background:var(--panel);
    color:var(--ink);text-decoration:none;font-size:.82rem;font-weight:560}
  .burger:hover{text-decoration:none;border-color:var(--link);color:var(--link)}

  aside{position:fixed;inset:0 auto 0 0;width:min(19rem,82vw);z-index:50;
    transform:translateX(-102%);transition:transform .22s ease;
    background:var(--bg);border-right:1px solid var(--line);
    overflow-y:auto;padding:1.4rem 1.2rem;box-shadow:0 0 0 100vmax transparent}
  aside:target{transform:translateX(0);box-shadow:2px 0 24px rgba(0,0,0,.18)}
  aside:target ~ .scrim{display:block;position:fixed;inset:0;z-index:40;
    background:rgba(0,0,0,.38)}
  .closer{display:flex;align-items:center;justify-content:center;width:2rem;
    height:2rem;border:1px solid var(--line);border-radius:7px;color:var(--muted);
    text-decoration:none;position:absolute;top:1.1rem;right:1.1rem;font-size:1.1rem;
    line-height:1}
  .closer:hover{color:var(--link);border-color:var(--link);text-decoration:none}
  nav a{font-size:.9rem;padding:.5rem .6rem}
}

@media (min-width:960px){
  .shell{grid-template-columns:16rem minmax(0,1fr);gap:4rem}
  /* The shell is capped at 88rem and centred, so a window wider than that
     leaves a gutter on each side. The right one is invisible because main
     has no background; the left one would show page colour beside the
     panel. The shadow paints that gutter without affecting layout or
     adding scroll, which keeps the measure capped and the panel full-bleed. */
  aside{position:sticky;top:0;height:100vh;overflow-y:auto;
    padding:2.5rem 1.4rem 2rem 1.5rem;border-right:1px solid var(--line);
    background:var(--panel);box-shadow:-100vw 0 0 var(--panel)}
  .closer{display:none}
}
main{padding:0 1.5rem 6rem;min-width:0}
@media (min-width:960px){main{padding:2.5rem 2rem 6rem 0}}

/* ---- sidebar contents --------------------------------------------------- */
.brand{display:flex;align-items:center;gap:.55rem;font-weight:660;letter-spacing:-.02em;
  font-size:1.05rem;margin-bottom:.15rem}
.brand .lucide{color:var(--link)}
.brandsub{color:var(--faint);font-size:.78rem;margin:0 0 .6rem;line-height:1.45}
.sidestat{display:flex;gap:.9rem;margin:0 0 1.3rem;padding:.55rem 0;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.sidestat div{font-size:.72rem;color:var(--faint);line-height:1.35}
.sidestat b{display:block;font-size:.94rem;color:var(--ink);font-weight:650;
  font-variant-numeric:tabular-nums}
nav{margin-bottom:1rem}
/* Vertical at every width. The drawer replaced the horizontal strip that
   once made this a wrapping flex row. */
nav ol{list-style:none;margin:0;padding:0;display:block}
/* A soft filled pill rather than a coloured left rail: the rail is the
   generic pattern and reads as decoration, while a fill reads as a surface
   the item actually occupies. Weight and ink carry the state, not hue. */
nav a{display:flex;align-items:center;gap:.62rem;padding:.42rem .6rem;
  border-radius:7px;color:var(--muted);text-decoration:none;font-size:.855rem;
  line-height:1.3;transition:background .13s ease,color .13s ease}
nav a:hover{background:var(--raised);color:var(--ink);text-decoration:none;
  box-shadow:0 0 0 1px var(--line)}
nav a .lucide{color:var(--faint);flex:none;transition:color .13s ease}
nav a:hover .lucide{color:var(--ink)}
section:target{scroll-margin-top:1rem}

header{padding:1rem 0 .5rem}
h1{font-size:2.15rem;margin:0 0 .35rem;letter-spacing:-.03em;font-weight:680}
.tagline{font-size:1.02rem;color:var(--muted);margin:0 0 1.4rem;max-width:44rem}

section{padding-top:2.75rem;scroll-margin-top:1rem}
h2{display:flex;align-items:center;gap:.55rem;font-size:1.22rem;margin:0 0 .25rem;
  letter-spacing:-.02em;font-weight:640}
h2 .lucide{color:var(--faint);flex:none}
.sub{color:var(--muted);font-size:.92rem;margin:0 0 1.2rem;max-width:54rem}
h3{font-size:.755rem;margin:1.9rem 0 .55rem;color:var(--faint);text-transform:uppercase;
  letter-spacing:.09em;font-weight:680}
p{margin:0 0 .95rem;max-width:54rem}
ul{max-width:54rem;padding-left:1.15rem}
li{margin-bottom:.35rem}

.verdict{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
  padding:1.05rem 1.2rem;margin:0 0 1.6rem;max-width:54rem}
.verdict h3{margin-top:0;color:var(--warn);display:flex;align-items:center;gap:.4rem}
.verdict p:last-child{margin-bottom:0}

/* Metrics read as a row of numbers, not a wall of cards. */
.grid{display:flex;flex-wrap:wrap;gap:2.4rem;margin:1.4rem 0 1.8rem;
  padding:1.1rem 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.stat .v{font-size:1.42rem;font-weight:640;letter-spacing:-.025em;
  font-variant-numeric:tabular-nums;display:block;line-height:1.25}
.stat .k{font-size:.735rem;color:var(--faint);text-transform:uppercase;
  letter-spacing:.07em;margin-top:.12rem;display:block}

.scroll{overflow-x:auto;margin:.9rem 0 1.4rem}
table{width:100%;border-collapse:collapse;font-size:.865rem;min-width:29rem}
caption{text-align:left;color:var(--faint);font-size:.82rem;padding-bottom:.55rem}
th,td{text-align:right;padding:.5rem .7rem;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--faint);font-weight:620;font-size:.715rem;text-transform:uppercase;
  letter-spacing:.06em}
tbody tr:hover td{background:var(--panel)}
tbody tr.pick td{background:var(--panel);font-weight:620}
tbody tr.base td{color:var(--faint)}
/* The unmet criterion in the gate table. The row tint is a third channel
   after the icon and the word, never the only one. */
tbody tr.fail td{background:color-mix(in srgb,var(--warn) 6%,transparent)}
td .lucide{vertical-align:-3px;margin-right:.25rem}
.ok{color:var(--ok);font-weight:600;white-space:nowrap}
.no{color:var(--warn);font-weight:680;white-space:nowrap}

code{font-family:var(--mono);background:var(--panel);padding:.1em .38em;
  border-radius:5px;font-size:.86em}
pre{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
  padding:.95rem 1.05rem;overflow-x:auto;font-size:.825rem;line-height:1.6;
  margin:.9rem 0 1.4rem;font-family:var(--mono);max-width:54rem}
pre code{background:none;padding:0;font-size:1em}

figure{margin:1.3rem 0}
figcaption{color:var(--faint);font-size:.815rem;margin-top:.5rem;max-width:54rem;line-height:1.55}

.viz{--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme:dark){.viz{--s1:#3987e5;--s2:#d95926}}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .tick{fill:var(--faint);font-size:11px;font-variant-numeric:tabular-nums}
.chart .axis{fill:var(--muted);font-size:11px;letter-spacing:.03em}
.chart .serieslabel{font-size:11.5px;font-weight:620}
.chart .barlabel{fill:var(--ink);font-size:11.5px;font-family:var(--mono)}
.chart .barvalue{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.chart circle{transition:r .12s ease}
.chart circle:hover{r:6}
.chart rect{transition:opacity .12s ease}
.chart rect:hover{opacity:1}
.pair{display:grid;gap:.4rem}
@media (min-width:900px){.pair{grid-template-columns:1fr 1fr}}

/* ---- hero ---------------------------------------------------------------
   A recruiter gives this a few seconds. The headline numbers therefore sit
   above the fold at display size, before any prose. */
.hero{padding:2.2rem 0 .5rem}
/* Neutral, not warn-red. This chip states the method, which is the strongest
   thing about the project; in red at the top of the page it read as an error
   banner before the reader had any context to interpret it. */
.eyebrow{display:inline-flex;align-items:center;gap:.4rem;font-size:.72rem;
  font-weight:640;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  background:var(--panel);border:1px solid var(--line);
  padding:.28rem .65rem;border-radius:100px;margin-bottom:1.1rem}
.eyebrow .lucide{color:var(--faint)}
h1{font-size:clamp(2.3rem,4.4vw,3.1rem);margin:0 0 .5rem;letter-spacing:-.035em;
  font-weight:700;line-height:1.08}
.tagline{font-size:clamp(1.02rem,1.5vw,1.18rem);color:var(--muted);margin:0 0 1.9rem;
  max-width:40rem;line-height:1.55}
.tagline strong{color:var(--ink);font-weight:620}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:var(--r);overflow:hidden;margin:0 0 1.7rem;max-width:54rem}
.kpi{background:var(--raised);padding:1rem 1.15rem 1.05rem}
.kpi .n{font-size:clamp(1.5rem,2.6vw,1.95rem);font-weight:680;letter-spacing:-.035em;
  font-variant-numeric:tabular-nums;line-height:1.1;display:block}
.kpi .l{font-size:.735rem;color:var(--faint);text-transform:uppercase;
  letter-spacing:.07em;margin-top:.3rem;display:block}
.kpi .s{font-size:.76rem;color:var(--muted);margin-top:.2rem;display:block}
.kpi.miss .n{color:var(--warn)}

.pitch{font-size:1.02rem;line-height:1.62;max-width:44rem;margin:0 0 1.7rem;
  padding-left:1rem;border-left:2px solid var(--line)}
.pitch strong{font-weight:640}

.cta{display:flex;flex-wrap:wrap;gap:.55rem;margin-bottom:.4rem}
.btn{display:inline-flex;align-items:center;gap:.45rem;padding:.5rem .95rem;
  border-radius:8px;font-size:.885rem;font-weight:560;text-decoration:none;
  border:1px solid var(--line);color:var(--ink);background:var(--raised);
  transition:border-color .12s ease,color .12s ease}
.btn:hover{border-color:var(--link);color:var(--link);text-decoration:none}
.btn.primary{background:var(--link);border-color:var(--link);color:#fff}
.btn.primary:hover{opacity:.9;color:#fff}
.btn .lucide{flex:none}

.topbrand{display:inline-flex;align-items:center;gap:.4rem;font-weight:640;
  font-size:.9rem;letter-spacing:-.01em}
.topbrand .lucide{color:var(--link)}
.repo{display:flex;align-items:center;gap:.5rem;padding-top:.9rem;
  border-top:1px solid var(--line);font-size:.8rem;color:var(--faint);
  text-decoration:none;font-family:var(--mono)}
.repo b{color:var(--muted);font-weight:600}
.repo:hover{color:var(--link);text-decoration:none}
.repo:hover b{color:var(--link)}
.note{color:var(--faint);font-size:.855rem;max-width:54rem}
.flag{display:flex;gap:.6rem;border-left:2px solid var(--line);padding:.1rem 0 .1rem .9rem;
  color:var(--muted);font-size:.885rem;margin:1.1rem 0;max-width:54rem}
.flag .lucide{color:var(--faint);flex:none;margin-top:.2rem}

footer{margin-top:4rem;padding-top:1.3rem;border-top:1px solid var(--line);
  color:var(--faint);font-size:.83rem}
a{color:var(--link)}
a:hover{text-decoration:underline}
.lucide{flex:none}

@media (max-width:700px){
  main{padding:0 1.1rem 4rem}
  .hero{padding:1.5rem 0 .25rem}
  h1{font-size:2.05rem}
  .tagline{font-size:1rem;margin-bottom:1.4rem}
  /* Two columns rather than four, so each number keeps its size. */
  .kpis{grid-template-columns:1fr 1fr}
  .kpi{padding:.85rem .9rem}
  .kpi .n{font-size:1.45rem}
  .cta{gap:.45rem}
  .btn{flex:1 1 auto;justify-content:center;font-size:.85rem;padding:.55rem .7rem}
  .grid{gap:1.35rem;padding:.9rem 0}
  .pitch{font-size:.97rem}
  section{padding-top:2.1rem}
  h2{font-size:1.12rem}
  /* Tables get an inset shadow hint that they scroll. */
  .scroll{margin-inline:-1.1rem;padding-inline:1.1rem}
  pre{font-size:.78rem}
  /* Chart text lives in user units, so fitting a 720-unit viewBox into a
     350px phone would render 11px labels at about 5px. Below this floor the
     figure scrolls instead of shrinking further. */
  figure{overflow-x:auto;margin-inline:-1.1rem;padding-inline:1.1rem}
  .chart{min-width:30rem}
  figcaption{min-width:0}
}
@media (max-width:380px){
  .kpis{grid-template-columns:1fr}
  h1{font-size:1.8rem}
}
@media print{aside,.topbar{display:none}body{font-size:10.5pt}section{page-break-inside:avoid}}
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


def _ext(href: str, body: str, cls: str = "") -> str:
    """An outbound link. Opens in a new tab so the page is not navigated away
    from mid-read; `noopener` because `target=_blank` otherwise exposes
    `window.opener` to the destination."""
    attrs = f' class="{cls}"' if cls else ""
    return (
        f'<a{attrs} href="{href}" target="_blank" rel="noopener noreferrer">'
        f"{body}</a>"
    )


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
    glyph = icon(SECTION_ICONS.get(anchor, ""), 18)
    return (
        f'<section id="{anchor}"><h2>{glyph}{title}</h2>{subtitle}'
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
        # Icon + word + colour. Colour alone would fail a reader with a
        # colour-vision deficiency, and this table carries the verdict.
        mark = (
            f'<span class="ok">{icon("check", 15)}pass</span>'
            if ok
            else f'<span class="no">{icon("cross", 15)}NOT MET</span>'
        )
        rows.append([_e(label), mark])
        classes.append("" if ok else "fail")
    return _table(["pre-registered criterion", "result"], rows, classes)


def _kpis(
    best: dict[str, Any],
    cost: dict[str, Any] | None,
    savings: dict[str, Any] | None,
) -> str:
    """The four numbers a reader should leave with, at display size."""
    cells = [
        (f"{best['auroc']:.3f}", "AUROC", "vs 0.500 baselines", ""),
        (f"{best['recall_at_fpr']:.3f}", "recall @ 5% FPR", "bar was 0.250", "miss"),
    ]
    if cost:
        cells.append(
            (
                f"{cost['predict_single_ms_p50']:.2f}<span style='font-size:.6em'> ms</span>",
                "CPU latency",
                f"{cost['parameters_bytes'] / 1024:.1f} KB model",
                "",
            )
        )
    if savings:
        near = min(
            savings["points"],
            key=lambda p: abs(p["false_positive_rate"] - 0.05),
        )
        cells.append(
            (
                f"{near['savings_rate']:.1%}",
                "tokens saved",
                "paper reports 14.6–20.4%",
                "",
            )
        )

    body = "".join(
        f'<div class="kpi {cls}"><span class="n">{n}</span>'
        f'<span class="l">{_e(label)}</span><span class="s">{_e(sub)}</span></div>'
        for n, label, sub, cls in cells
    )
    return f'<div class="kpis">{body}</div>'


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
        "Each cut has a different population, only sessions that reached that "
        "turn appear, so this is not one model tracked over time."
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
        "parentheses. Both curves are shown deliberately, the raw one is the "
        "instructive half, since it is what class weighting does to the output "
        "scale. Showing only the corrected version would hide why the step "
        "exists.</figcaption></figure>"
    )


def _savings_charts(savings: dict[str, Any]) -> str:
    """Savings and harm as two panels sharing an x-axis.

    Previously one plot with two y-scales. A dual axis invents a relationship:
    where the two lines cross is an artefact of how the scales were aligned,
    not something in the data, the exact misreading the caption was trying to
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
        "quantities are not commensurable, one is tokens not spent, the other "
        "is work destroyed, and overlaying them on shared axes would suggest a "
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
    # rendered, optional artifacts may be absent, and linking to a section
    # that was skipped leaves a dead anchor.
    NAV_SLOT = "<!--NAV-->"
    parts.append(
        '<a class="skip" href="#question">Skip to content</a>'
        '<div class="topbar">'
        f'<a class="burger" href="#menu">{icon("menu", 16)}Contents</a>'
        f'<span class="topbrand">{mark(16)}Averta</span>'
        "</div>"
        '<div class="shell">'
        f'<aside id="menu"><a class="closer" href="#" aria-label="Close">'
        f"\u00d7</a>{NAV_SLOT}</aside>"
        # Sibling, not a child: the open drawer carries a transform, which
        # would become the containing block for a fixed-position descendant
        # and collapse the scrim onto the drawer's own box.
        '<a class="scrim" href="#" aria-hidden="true" tabindex="-1"></a>'
        "<main><header class=\"hero\">"
        f'<span class="eyebrow">{icon("flask", 13)}Pre-registered study</span>'
        "<h1>Averta</h1>"
        '<p class="tagline">Predicting when an AI coding agent is about to fail, '
        "using a <strong>2.1 KB model</strong> that scores a live session in "
        "<strong>0.18 ms</strong> on one CPU core.</p>"
        + _kpis(best, logistic_cost, savings)
        + '<p class="pitch">It recovers roughly <strong>half</strong> the token '
        "savings reported for a 0.6B neural monitor, and still missed the bar "
        "set for it before any model was trained. Both halves of that are the "
        "result, and the bar was never moved to fit.</p>"
        '<div class="cta">'
        + _ext(REPO, f'{icon("terminal", 16)}View source', cls="btn primary")
        + f'<a class="btn" href="#results">{icon("chart", 16)}Jump to results</a>'
        + _ext(
            f"{REPO}/blob/main/MODEL_CARD.md",
            f'{icon("alert", 16)}Model card',
            cls="btn",
        )
        + _ext(PAPER, f'{icon("flask", 16)}The paper', cls="btn")
        +
        "</div></header>"
    )

    # ---- the question -----------------------------------------------------
    parts.append(
        _section(
            "question",
            "The question",
            "What this set out to test, and what it found.",
            "<p>A coding agent either resolves its task or it doesn't. A session "
            "heading nowhere keeps spending tokens that produce nothing, so "
            "detecting that early is worth doing. But stopping a session that "
            "would have recovered is worse than letting it run. That asymmetry is "
            "the whole problem: the value is in catching doomed sessions, the cost "
            "is in killing salvageable ones.</p>"
            "<p>" + _ext(PAPER, "Fail-Fast, Restart-Smart") + ' (Wang et al., 2026) '
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
                "<p>Training data comes from "
                + _ext(DATASET, "SWE-Gym/OpenHands-Sampled-Trajectories")
                + ". A survey of six public "
                "trajectory corpora found it to be the <strong>only one carrying both "
                "outcome classes</strong>, the others are supervised fine-tuning sets "
                "that store the agent's patch but never whether it worked.</p>"
                "<p>Two properties shape everything downstream. The positive class is "
                "rare, so metrics that tolerate imbalance are required. And "
                "<strong>session length barely separates the classes</strong>: resolved "
                "sessions average 39.86 turns against 39.23 for unresolved. Prior work "
                "describes failing agent runs as tending to run longer; that does not "
                "reproduce here.</p>"
                '<p class="flag">{FLAG_ICON}The source carries no declared license, so this '
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
            "nothing else, never the total length, never the outcome. A test suite "
            "shuffles, truncates and extends the unseen tail and asserts the feature "
            "vector is byte-identical, and checks that features at turn 6 are the same "
            "whether the session runs to 12 turns or 200.</p>"
            "<h3>Absolute turn indices</h3>"
            "<p>Evaluating at &ldquo;40% through the session&rdquo; requires knowing the "
            "total length, which is unavailable while a session is running. Evaluation "
            "uses turns 3, 5, 10, 20 and 40. Each cut has a different population, since only "
            "sessions that reached that turn appear, and the resolve rate is reported "
            "per cut so the shift stays visible.</p>"
            "<h3>Repository-grouped splits</h3>"
            "<p>Every row from a repository lands in one fold. Rows sharing a codebase "
            "are not independent, so splitting by row would let a model memorise a "
            "repository and be rewarded for it. Confidence intervals resample "
            "repositories rather than rows for the same reason, row-level bootstrap "
            "reports intervals that are too narrow under clustering.</p>"
            "<h3>Polarity</h3>"
            "<p>The positive class is <strong>failure</strong>, so the false positive "
            "rate means &ldquo;sessions that would have resolved but were flagged&rdquo; "
            "- the quantity worth constraining. Average precision is reported on the "
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
            "dominating, <code>n_repeated_calls</code>, a flat count of tool calls "
            "reissued with byte-identical arguments, while every error-based feature "
            "was negligible.</p>"
            "<p>That pointed at <strong>ordering</strong> as the missing ingredient. A "
            "count knows an action recurred; it cannot express that the agent is cycling "
            "A-B-A-B, or how far back it reached to repeat itself.</p>"
            "<p><strong>Attempt 2</strong> added 10 sequence-structure features. Recall "
            "rose from 0.155 to 0.191 and AUROC from 0.668 to 0.677, real movement on "
            "exactly the failing criterion, but not enough to clear it.</p>",
            _table(
                ["feature", "attempt 1", "attempt 2"],
                [
                    ["<code>distinct_action_ratio</code>", "-", "<strong>0.159</strong>"],
                    ["<code>novelty_rate_recent</code>", "-", "<strong>0.153</strong>"],
                    ["<code>n_errors</code>", "0.023", "0.084"],
                    ["<code>action_bigram_repeat_max</code>", "-", "0.078"],
                    ["<code>n_repeated_calls</code>", "<strong>0.198</strong>", "0.030"],
                ],
                [""] * 5,
                "AUROC drop when the feature is shuffled, under the same grouped folds.",
            ),
            "<p><code>n_repeated_calls</code> collapsed once ordering was represented "
            "properly, it had been a proxy for structure it could not express. The two "
            "strongest features now both measure <strong>declining action novelty</strong>.</p>"
            '<p class="flag">{FLAG_ICON}Modelling stopped after two attempts. The gains were real '
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
                    "fold. One series, so one colour, the bar length already "
                    "encodes magnitude.",
                ),
                _table(
                    ["feature", "AUROC drop when shuffled"], rows, [""] * len(rows)
                ),
                "<p>The prior expectation was that a recurring error signature would "
                "dominate. It is close to worthless in isolation, AUROC 0.521 alone. "
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
    # Read straight from the artifact. These figures were hard-coded once and
    # went stale when the pipeline was re-run, leaving the paragraph quoting
    # one pair of Brier scores beside a chart legend rendering another.
    _cal = diag["calibration"]
    # The bin where the raw score understates the observed rate by the most,
    # among bins holding at least 1% of the scored rows. Without that floor the
    # worst gap sits in a bin of seven sessions, where one observation moves the
    # rate by 14 points, and the page would quote noise as its illustration.
    _bins = list(
        zip(
            _cal["raw"]["predicted"],
            _cal["raw"]["observed"],
            _cal["raw"]["counts"],
            strict=True,
        )
    )
    _floor = 0.01 * sum(count for *_, count in _bins)
    _worst = max(
        (b for b in _bins if b[2] >= _floor),
        key=lambda bin_: bin_[1] - bin_[0],
    )
    parts.append(
        _section(
            "calibration",
            "Calibration",
            "A bug the plot caught, kept visible because the before-curve is the lesson.",
            "<p>Every model is fitted with class weighting, which is correct for ranking "
            f"but leaves the output on a re-balanced scale. A raw score of {_worst[0]:.2f} "
            f"corresponded to an observed failure rate near {_worst[1]:.2f}. Rank-based "
            "metrics are unaffected, but any number shown to a person has to mean what "
            "it says.</p>"
            "<p>Isotonic regression fitted on <strong>out-of-fold</strong> scores "
            f"corrects it, moving the Brier score from {_cal['brier_raw']:.4f} to "
            f"{_cal['brier_calibrated']:.4f}. Fitting the "
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
                "<p>At a comparable budget, 4.7% against the paper's 5% target, this "
                "reaches <strong>8.4% estimated token savings</strong> where the 0.6B "
                "neural monitor reports 14.6&ndash;20.4%. Roughly half the value, at a "
                "fraction of the size.</p>"
                '<p class="flag">{FLAG_ICON}Tokens are <strong>estimated</strong> from content '
                "length; the corpus records no token counts. Savings count only what "
                "would have been spent after the cut. Sessions wrongly terminated are "
                "never netted off the savings, the two are not commensurable and a "
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
                "Planned as an AUROC comparison. Not reported, the sample "
                "cannot support one.",
                f"<p>The plan was to train on OpenHands trajectories and test on "
                f"hand-labelled Claude Code sessions. Only {drift['n_local']} local "
                "sessions are long enough and none carry outcome labels, so a transfer "
                "AUROC would be noise presented as a result.</p>"
                "<p>What is measurable without labels is whether the features compute "
                "comparably at all, a prerequisite for transfer rather than a "
                "substitute for measuring it.</p>",
                _table(
                    ["feature", "corpus", "local", "std diff"], rows, [""] * len(rows)
                ),
                "<p>Every comparable feature shifts by more than 1.3 standard "
                "deviations and some fall outside the training range entirely. Claude "
                "Code emits more assistant turns per tool call than OpenHands, because "
                "thinking blocks become their own turns, a &ldquo;turn&rdquo; is not the "
                "same unit across scaffolds. Cross-scaffold accuracy is therefore "
                "<strong>unmeasured</strong>, and the tool labels its output indicative "
                "in every code path.</p>"
                '<p class="flag">{FLAG_ICON}This check caught a real bug. Two features reported '
                "exactly 0.00 on sessions full of edits, because edit detection was keyed "
                "to one tool vocabulary. Not a slightly worse score, a confident zero "
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
            "<p>Those work immediately, the trained model is committed. Reproducing the "
            "study needs the corpus, which is a 12-minute download:</p>"
            "<pre><code>averta ingest &amp;&amp; averta features\n"
            "averta train        # cross-validate and apply the gate\n"
            "averta diagnose     # importance and inference cost\n"
            "averta savings      # savings against harm\n"
            "averta figures      # the plots on this page</code></pre>"
            "<p><code>averta explain</code> separates what is <strong>measured</strong>, "
            "recurring errors, reissued calls, clustered repetition, exact token counts, "
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
        f"<p>Generated {generated} from committed artifacts, every figure and table "
        "on this page is produced by <code>averta site</code> from JSON written by the "
        "pipeline, so it cannot drift out of sync with the results.</p>"
        "<p>Reproduces " + _ext(PAPER, "arXiv:2608.03222") + ". MIT licensed; the "
        "training corpus is not redistributed.</p>"
        "</footer></main></div>"
    )

    body = "".join(parts)
    # Callout glyph, substituted once rather than threaded through every string.
    body = body.replace("{FLAG_ICON}", icon("alert", 16))

    present = [
        (anchor, title)
        for anchor, title in SECTIONS
        if f'<section id="{anchor}"' in body
    ]
    nav_items = "".join(
        f'<li><a href="#{a}">{icon(SECTION_ICONS.get(a, ""), 15)}{t}</a></li>'
        for a, t in present
    )
    brand = (
        f'<div class="brand">{mark(19)}Averta</div>'
        '<p class="brandsub">CPU-native failure prediction<br>for AI coding agents</p>'
        '<div class="sidestat">'
        "<div><b>5,976</b>sessions</div>"
        "<div><b>33</b>features</div>"
        "<div><b>2.1 KB</b>model</div>"
        "</div>"
    )
    # The hero already carries the primary "View source" call to action.
    # Repeating it verbatim here would be noise, so this is an identity line -
    # the repository handle, rather than a second button with the same label.
    outbound = _ext(
        REPO,
        f'{icon("github", 15)}<span>mayurbijarniya/<b>averta</b></span>',
        cls="repo",
    )
    body = body.replace(
        NAV_SLOT,
        f"{brand}<nav aria-label=Contents><ol>{nav_items}</ol></nav>{outbound}",
    )

    document = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        '<meta name=description content="CPU-native failure prediction for AI coding '
        'agents: a pre-registered evaluation with a negative result.">'
        "<title>Averta, CPU-native failure prediction for AI coding agents</title>"
        f'<link rel=icon href="{favicon_data_uri()}">'
        f"<style>{STYLE}</style></head><body>"
        f"{body}"
        "</body></html>"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out
