"""Lucide icons, inlined.

Lucide ships MIT-licensed SVG paths, so the icons can be embedded directly
rather than pulled from a package. That keeps the page dependency-free while
still using the same icon set a React build would.

Every icon here is drawn at 24×24 on Lucide's grid with a 2px stroke and
`currentColor`, so it inherits text colour and themes with the page. Icons
always sit beside a text label — never alone, and never as the only carrier of
meaning.
"""

from __future__ import annotations

# Path data taken from lucide.dev, unmodified. Stroke attributes are applied by
# the wrapper so every icon renders consistently.
_PATHS: dict[str, str] = {
    # flask-conical — the study itself
    "flask": (
        '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1'
        '-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.453 15h11.094"/>'
        '<path d="M8.5 2h7"/>'
    ),
    # database — corpus
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
        '<path d="M3 12A9 3 0 0 0 21 12"/>'
    ),
    # sliders-horizontal — method
    "sliders": (
        '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/>'
        '<line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/>'
        '<line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/>'
        '<line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/>'
        '<line x1="16" x2="16" y1="18" y2="22"/>'
    ),
    # chart-no-axes-combined — results
    "chart": (
        '<path d="M12 16v5"/><path d="M16 14v7"/><path d="M20 10v11"/>'
        '<path d="m22 3-8.646 8.646a.5.5 0 0 1-.708 0L9.354 8.354a.5.5 0 0 0-.707 0L2 15"/>'
        '<path d="M4 18v3"/><path d="M8 14v7"/>'
    ),
    # git-branch — the two attempts
    "branch": (
        '<line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/>'
        '<circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>'
    ),
    # gauge — what drives it
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    # timer — inference cost
    "timer": (
        '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="15" y1="14" y2="11"/>'
        '<circle cx="12" cy="14" r="8"/>'
    ),
    # scale — calibration
    "scale": (
        '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>'
    ),
    # circle-dollar-sign — savings
    "cost": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8"/><path d="M12 18V6"/>'
    ),
    # shuffle — cross-scaffold transfer
    "shuffle": (
        '<path d="m18 14 4 4-4 4"/><path d="m18 2 4 4-4 4"/>'
        '<path d="M2 18h1.973a4 4 0 0 0 3.3-1.7l5.454-8.6a4 4 0 0 1 3.3-1.7H22"/>'
        '<path d="M2 6h1.972a4 4 0 0 1 3.6 2.2"/>'
        '<path d="M22 18h-6.041a4 4 0 0 1-3.3-1.8l-.359-.45"/>'
    ),
    # triangle-alert — limitations
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    # terminal — using it
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
    # circle-check — a passing criterion
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    # circle-x — a failing criterion
    "cross": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    # cpu — the model itself
    "cpu": (
        '<path d="M12 20v2"/><path d="M12 2v2"/><path d="M17 20v2"/><path d="M17 2v2"/>'
        '<path d="M2 12h2"/><path d="M2 17h2"/><path d="M2 7h2"/><path d="M20 12h2"/>'
        '<path d="M20 17h2"/><path d="M20 7h2"/><path d="M7 20v2"/><path d="M7 2v2"/>'
        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
        '<rect x="8" y="8" width="8" height="8" rx="1"/>'
    ),
}


def icon(name: str, size: int = 17, extra_class: str = "") -> str:
    """Inline one Lucide icon at the given pixel size."""
    path = _PATHS.get(name)
    if path is None:
        return ""
    classes = f"lucide {extra_class}".strip()
    return (
        f'<svg class="{classes}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{path}</svg>'
    )


SECTION_ICONS = {
    "question": "flask",
    "data": "database",
    "method": "sliders",
    "results": "chart",
    "attempts": "branch",
    "drivers": "gauge",
    "cost": "timer",
    "calibration": "scale",
    "savings": "cost",
    "transfer": "shuffle",
    "limitations": "alert",
    "using": "terminal",
}
