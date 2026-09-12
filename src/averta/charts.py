"""Inline SVG charts for the results page.

Replaces rendered PNGs for three reasons. Matplotlib bakes a white background
into the file, so on a dark page the figures were three white rectangles.
Raster images blur when zoomed. And a PNG cannot respond to
`prefers-color-scheme`, while an SVG that paints with `currentColor` and CSS
custom properties simply does.

No JavaScript. Hover affordances are CSS-only, so the page still works from a
`file://` URL with scripting disabled.

Colours are the validated categorical slots 1 (blue) and 2 (orange), which
clear the CVD, chroma, lightness and contrast checks in both light and dark
modes. Series are never distinguished by colour alone, every series carries a
direct label as well.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

# Validated categorical slots. Themed via CSS custom properties in site.py.
SERIES_1 = "var(--s1)"
SERIES_2 = "var(--s2)"
INK = "var(--ink)"
MUTED = "var(--muted)"
GRID = "var(--line)"


@dataclass(frozen=True)
class Series:
    label: str
    points: list[tuple[float, float]]
    color: str = SERIES_1
    dashed: bool = False
    band: list[tuple[float, float, float]] | None = None


def _e(text: object) -> str:
    return html.escape(str(text))


def _scale(lo: float, hi: float, size: float, pad_lo: float, pad_hi: float):
    span = hi - lo or 1.0
    return lambda v: pad_lo + (v - lo) / span * (size - pad_lo - pad_hi)


def line_chart(
    series: list[Series],
    *,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
    title: str,
    width: int = 720,
    height: int = 330,
    reference: tuple[float, str] | None = None,
    x_categorical: bool = False,
) -> str:
    """Line chart with optional confidence bands and a reference rule.

    `x_categorical` spaces the ticks evenly rather than by value, cut points
    of 3/5/10/20/40 are ordered labels, not a continuous axis, and spacing them
    linearly would imply the gap between 20 and 40 carries meaning.
    """
    left, right, top, bottom = 52, 118, 34, 46

    # Categorical axes place ticks at even positions; continuous axes map the
    # value straight through, so points need not coincide with a labelled tick.
    if x_categorical:
        index_of = {t: i for i, t in enumerate(x_ticks)}
        x_of_value = lambda v: index_of[v]  # noqa: E731
        domain = [float(i) for i in range(len(x_ticks))]
    else:
        x_of_value = lambda v: v  # noqa: E731
        domain = [v for item in series for v, _ in item.points] + list(x_ticks)

    sx = _scale(min(domain), max(domain), width, left, right)
    sy = _scale(min(y_ticks), max(y_ticks), height, bottom, top)
    py = lambda v: height - sy(v)  # noqa: E731, flip to screen coordinates

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_e(title)}" xmlns="http://www.w3.org/2000/svg">',
        f'<title>{_e(title)}</title>',
    ]

    # Hairline grid, one shade off the surface.
    for t in y_ticks:
        y = py(t)
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{t:g}</text>'
        )

    for t in x_ticks:
        x = sx(x_of_value(t))
        out.append(
            f'<text x="{x:.1f}" y="{height - bottom + 20}" text-anchor="middle" '
            f'class="tick">{t:g}</text>'
        )

    if reference is not None:
        value, caption = reference
        y = py(value)
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 3"/>'
        )
        out.append(
            f'<text x="{width - right + 6}" y="{y + 4:.1f}" class="tick">'
            f"{_e(caption)}</text>"
        )

    for item in series:
        if item.band:
            upper = " ".join(
                f"{sx(x_of_value(x)):.1f},{py(hi):.1f}" for x, _, hi in item.band
            )
            lower = " ".join(
                f"{sx(x_of_value(x)):.1f},{py(lo):.1f}"
                for x, lo, _ in reversed(item.band)
            )
            out.append(
                f'<polygon points="{upper} {lower}" fill="{item.color}" '
                'fill-opacity="0.13"/>'
            )

    for item in series:
        path = " ".join(
            f"{sx(x_of_value(x)):.1f},{py(y):.1f}" for x, y in item.points
        )
        dash = ' stroke-dasharray="4 4"' if item.dashed else ""
        out.append(
            f'<polyline points="{path}" fill="none" stroke="{item.color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
        for x, y in item.points:
            out.append(
                f'<circle cx="{sx(x_of_value(x)):.1f}" cy="{py(y):.1f}" r="4" '
                f'fill="{item.color}" stroke="var(--bg)" stroke-width="2">'
                f"<title>{_e(item.label)}: {y:.3f} at {x:g}</title></circle>"
            )
        # Direct label at the endpoint, so identity never rests on colour alone.
        last_x, last_y = item.points[-1]
        out.append(
            f'<text x="{sx(x_of_value(last_x)) + 9:.1f}" y="{py(last_y) + 4:.1f}" '
            f'class="serieslabel" fill="{item.color}">{_e(item.label)}</text>'
        )

    out.append(
        f'<text x="{left}" y="{height - 6}" class="axis">{_e(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(13,{top + 6}) rotate(-90)" '
        f'class="axis" text-anchor="end">{_e(y_label)}</text>'
    )
    out.append("</svg>")
    return "".join(out)


def bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    value_format: str = "{:.3f}",
    color: str = SERIES_1,
    width: int = 720,
    bar_height: int = 26,
    gap: int = 8,
    highlight: int | None = None,
) -> str:
    """Horizontal bars. Values are direct-labelled, so no value axis is drawn."""
    left = 230
    right = 74
    top = 12
    height = top * 2 + len(labels) * (bar_height + gap) - gap
    span = max(values) or 1.0
    usable = width - left - right

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_e(title)}" xmlns="http://www.w3.org/2000/svg">',
        f"<title>{_e(title)}</title>",
    ]

    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = top + index * (bar_height + gap)
        length = max(value / span * usable, 2)
        emphasis = "" if highlight is None or index == highlight else ' opacity="0.55"'
        out.append(
            f'<text x="{left - 12}" y="{y + bar_height / 2 + 4:.0f}" '
            f'text-anchor="end" class="barlabel">{_e(label)}</text>'
        )
        # 4px rounded data-end, anchored to the baseline.
        out.append(
            f'<rect x="{left}" y="{y}" width="{length:.1f}" height="{bar_height}" '
            f'rx="4" fill="{color}"{emphasis}><title>{_e(label)}: '
            f'{_e(value_format.format(value))}</title></rect>'
        )
        out.append(
            f'<text x="{left + length + 9:.1f}" y="{y + bar_height / 2 + 4:.0f}" '
            f'class="barvalue">{_e(value_format.format(value))}</text>'
        )

    out.append("</svg>")
    return "".join(out)


def paired_charts(
    x: list[float],
    left_values: list[float],
    right_values: list[float],
    *,
    left_label: str,
    right_label: str,
    x_label: str,
    marker: tuple[float, str] | None = None,
) -> str:
    """Two stacked charts sharing an x-axis, replacing a dual-axis plot.

    A single plot with two y-scales invents a relationship: where the lines
    cross is an artefact of how the scales were aligned, not something in the
    data. Two panels on a shared x-axis show the same trade-off without
    implying one.
    """
    # One rounding, used for both the points and the ticks, rounding them
    # separately leaves the tick lookup with keys the points never match.
    xs = [round(v, 3) for v in x]

    # Label a subset: one tick per value would collide at this width.
    stride = max(len(xs) // 5, 1)
    labelled = sorted({*xs[::stride], xs[-1]})

    panels = []
    for values, label, color in (
        (left_values, left_label, SERIES_1),
        (right_values, right_label, SERIES_2),
    ):
        top = max(values) or 1.0
        step = top / 4
        ticks = [round(step * i, 1) for i in range(5)]
        panels.append(
            line_chart(
                [
                    Series(
                        label=label,
                        points=list(zip(xs, values, strict=True)),
                        color=color,
                    )
                ],
                x_ticks=labelled,
                y_ticks=ticks,
                x_label=x_label,
                y_label=label,
                title=f"{label} against {x_label}",
                width=560,
                height=250,
                reference=marker,
            )
        )
    return "".join(f"<div>{panel}</div>" for panel in panels)
