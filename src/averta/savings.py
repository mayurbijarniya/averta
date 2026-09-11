"""Token-saving simulation and the harm it trades against.

The corpus records no per-message token counts, so tokens are **estimated**
from content length at a fixed characters-per-token ratio. Every figure this
module produces is therefore an estimate and is labelled as one. Live Claude
Code sessions do carry real `usage` counts; those are exact and handled by the
adapter, not here.

The simulation is deliberately conservative. Stopping a session at turn *t*
saves only what was spent *after* turn *t* — the tokens already consumed are
gone either way, and a restart is not free.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

CHARS_PER_TOKEN = 4.0

TOKENS_QUERY = """
SELECT s.session_id,
       s.outcome,
       sum(t.content_chars) AS total_chars,
       sum(CASE WHEN t.turn_index < ? THEN t.content_chars ELSE 0 END) AS prefix_chars
FROM session s JOIN turn t USING (session_id)
GROUP BY s.session_id, s.outcome
"""


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    flagged: int
    true_positives: int
    false_positives: int
    recall: float
    false_positive_rate: float
    estimated_tokens_saved: int
    estimated_tokens_total: int
    successes_terminated: int

    @property
    def savings_rate(self) -> float:
        if self.estimated_tokens_total == 0:
            return 0.0
        return self.estimated_tokens_saved / self.estimated_tokens_total

    def as_dict(self) -> dict[str, float]:
        """Serializable form, including the computed rate.

        `vars()` omits properties, which silently dropped `savings_rate` from
        the written JSON and broke the page generator downstream.
        """
        return {**vars(self), "savings_rate": self.savings_rate}


def load_token_estimates(db: str, cut_point: int) -> dict[str, tuple[float, float]]:
    """Maps session id to (estimated tokens after the cut, estimated total)."""
    conn = duckdb.connect(db, read_only=True)
    try:
        rows = conn.execute(TOKENS_QUERY, [cut_point]).fetchall()
    finally:
        conn.close()

    estimates = {}
    for session_id, _outcome, total_chars, prefix_chars in rows:
        total = (total_chars or 0) / CHARS_PER_TOKEN
        prefix = (prefix_chars or 0) / CHARS_PER_TOKEN
        estimates[session_id] = (max(total - prefix, 0.0), total)
    return estimates


def simulate(
    session_ids: np.ndarray,
    y_fail: np.ndarray,
    p_fail: np.ndarray,
    estimates: dict[str, tuple[float, float]],
    thresholds: np.ndarray | None = None,
) -> list[OperatingPoint]:
    """Sweep intervention thresholds, reporting savings against harm.

    A flagged failing session saves the tokens it would have spent after the
    cut. A flagged *successful* session saves nothing and destroys the result —
    counted separately as `successes_terminated`, never netted off the savings.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)

    remaining = np.array([estimates.get(sid, (0.0, 0.0))[0] for sid in session_ids])
    totals = np.array([estimates.get(sid, (0.0, 0.0))[1] for sid in session_ids])

    total_tokens = float(totals.sum())
    negatives = int((y_fail == 0).sum())
    positives = int((y_fail == 1).sum())

    points = []
    for threshold in thresholds:
        flagged = p_fail >= threshold
        true_positive = flagged & (y_fail == 1)
        false_positive = flagged & (y_fail == 0)

        points.append(
            OperatingPoint(
                threshold=float(threshold),
                flagged=int(flagged.sum()),
                true_positives=int(true_positive.sum()),
                false_positives=int(false_positive.sum()),
                recall=float(true_positive.sum() / positives) if positives else 0.0,
                false_positive_rate=(
                    float(false_positive.sum() / negatives) if negatives else 0.0
                ),
                estimated_tokens_saved=int(remaining[true_positive].sum()),
                estimated_tokens_total=int(total_tokens),
                successes_terminated=int(false_positive.sum()),
            )
        )
    return points


def render(points: list[OperatingPoint]) -> str:
    header = (
        f"{'thresh':>7}{'flagged':>9}{'recall':>8}{'FPR':>7}"
        f"{'est. saved':>12}{'of total':>10}{'successes killed':>18}"
    )
    lines = [header, "-" * len(header)]
    for point in points:
        lines.append(
            f"{point.threshold:>7.2f}"
            f"{point.flagged:>9}"
            f"{point.recall:>8.3f}"
            f"{point.false_positive_rate:>7.3f}"
            f"{point.estimated_tokens_saved:>12,}"
            f"{point.savings_rate:>9.1%}"
            f"{point.successes_terminated:>18}"
        )
    lines.append("")
    lines.append(
        "Tokens are ESTIMATED from content length at "
        f"{CHARS_PER_TOKEN:.0f} chars/token; the corpus records no token counts."
    )
    lines.append(
        "Savings count only tokens spent after the cut point. Successes killed "
        "are never netted off savings."
    )
    return "\n".join(lines)
