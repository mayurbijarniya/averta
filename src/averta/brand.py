"""The Averta mark.

Drawn here rather than in `icons.py`, which carries Lucide's MIT-licensed
paths and should stay unmixed with original work.

The mark is the thesis as a glyph: a trajectory arrives, reaches a decision
point, and forks. One branch turns away and keeps going. The other continues
on its original heading and terminates in a dot, the session that was not
averted. It is the only icon on the page drawn at two weights, because a
favicon at 16px cannot carry the same detail as a 19px brand lockup.

Geometry sits on the same 24x24 grid and 2px round-capped stroke as the Lucide
set, so the inline version sits correctly beside text set in the icon rhythm.
"""

from __future__ import annotations

from urllib.parse import quote

# Light-mode brand blue, matching --s1 in the chart palette. Fixed rather than
# themed: a favicon is composited against browser chrome, not against the page.
BRAND_BLUE = "#2a78d6"

# 24-grid, inherits colour from the surrounding text.
_MARK_24 = (
    '<path d="M3 12h7"/>'
    '<path d="m10 12 8.5-5.5"/>'
    '<path d="m10 12 4.5 3.5"/>'
    '<circle cx="17.6" cy="17.6" r="1.7" fill="currentColor" stroke="none"/>'
)

# 32-grid, heavier stroke and wider spacing so the fork survives 16px.
_MARK_32 = (
    '<path d="M6.5 16h6.5"/>'
    '<path d="m13 16 11-7"/>'
    '<path d="m13 16 6 4.5"/>'
    '<circle cx="23.4" cy="24" r="2.2" fill="#fff" stroke="none"/>'
)


def mark(size: int = 19, extra_class: str = "") -> str:
    """The bare mark, inline, taking its colour from the current text colour."""
    classes = f"lucide {extra_class}".strip()
    return (
        f'<svg class="{classes}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{_MARK_24}</svg>'
    )


def favicon_svg() -> str:
    """The mark on a filled tile.

    A bare stroke mark disappears against browser chrome, which may be any
    colour and offers no contrast guarantee. The tile supplies its own.
    """
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        f'<rect width="32" height="32" rx="7" fill="{BRAND_BLUE}"/>'
        '<g fill="none" stroke="#fff" stroke-width="2.75" stroke-linecap="round" '
        f'stroke-linejoin="round">{_MARK_32}</g>'
        "</svg>"
    )


def favicon_data_uri() -> str:
    """The favicon as a data URI, so the page stays a single file."""
    return "data:image/svg+xml," + quote(favicon_svg(), safe="")
