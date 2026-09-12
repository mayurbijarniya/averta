"""Whole-session analysis: risk over time, and deterministic waste accounting.

Two halves with very different standing, kept visibly separate.

The **waste accounting** is exact. It counts recurring error signatures,
reissued tool calls and repeated edits to one path, all measured directly from
the transcript with no model involved. These numbers are as reliable as the
transcript itself.

The **risk trajectory** comes from a model that did not clear its gate and was
trained on a different agent scaffold. It is context, not evidence.

The trajectory stops at the largest evaluated cut point. Corpus sessions run to
about 100 turns while local sessions reach well over a thousand, so beyond that
cut every feature is far outside the fitted range and a curve there would be
extrapolation drawn as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from averta.features.view import TurnView
from averta.monitor import MIN_TURNS_TO_SCORE, Scorer


@dataclass(frozen=True)
class RiskPoint:
    turn: int
    probability: float


WINDOW = 25

MIN_CLUSTERED = 3


@dataclass(frozen=True)
class Repetition:
    """Something the session did repeatedly *within a short span*.

    Ranked by clustering, not by total count. Editing one file 27 times over a
    thousand turns is ordinary iterative work; running the same command four
    times in twelve turns is a loop. Ranking by raw totals surfaced the former
    and buried the latter.
    """

    kind: str
    label: str
    occurrences: int
    clustered: int
    first_turn: int
    last_turn: int
    cluster_start: int
    cluster_end: int
    chars_in_cluster: int

    @property
    def span(self) -> int:
        return self.last_turn - self.first_turn


@dataclass
class SessionReport:
    session_id: str
    turns: int
    trajectory: tuple[RiskPoint, ...] = ()
    repetitions: tuple[Repetition, ...] = field(default_factory=tuple)
    error_turns: int = 0
    user_rejections: int = 0
    turns_since_clean: int = 0
    chars_since_clean: int = 0
    scored_through: int | None = None

    @property
    def risk_direction(self) -> str:
        if len(self.trajectory) < 2:
            return "unknown"
        delta = self.trajectory[-1].probability - self.trajectory[0].probability
        if abs(delta) < 0.02:
            return "flat"
        return "rising" if delta > 0 else "falling"


def _sparkline(values: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return blocks[len(blocks) // 2] * len(values)
    return "".join(
        blocks[min(int((v - low) / (high - low) * len(blocks)), len(blocks) - 1)]
        for v in values
    )


def _repetitions(turns: list[TurnView]) -> list[Repetition]:
    """Everything the session repeated, ranked by how much followed the first."""
    found: list[Repetition] = []

    def collect(kind: str, key_of, label_of) -> None:
        positions: dict[str, list[int]] = {}
        for index, turn in enumerate(turns):
            key = key_of(turn)
            if key:
                positions.setdefault(key, []).append(index)

        for key, hits in positions.items():
            if len(hits) < MIN_CLUSTERED:
                continue

            # Densest run: the most occurrences falling inside any WINDOW turns.
            best_count, best_lo, best_hi = 0, hits[0], hits[0]
            start = 0
            for end in range(len(hits)):
                while hits[end] - hits[start] > WINDOW:
                    start += 1
                if end - start + 1 > best_count:
                    best_count = end - start + 1
                    best_lo, best_hi = hits[start], hits[end]

            if best_count < MIN_CLUSTERED:
                continue

            found.append(
                Repetition(
                    kind=kind,
                    label=label_of(key),
                    occurrences=len(hits),
                    clustered=best_count,
                    first_turn=turns[hits[0]].turn_index,
                    last_turn=turns[hits[-1]].turn_index,
                    cluster_start=turns[best_lo].turn_index,
                    cluster_end=turns[best_hi].turn_index,
                    chars_in_cluster=sum(
                        t.content_chars for t in turns[best_lo : best_hi + 1]
                    ),
                )
            )

    collect(
        "error",
        lambda t: t.error_signature,
        lambda key: key[:90],
    )
    collect(
        "tool call",
        lambda t: f"{t.tool_name}#{t.tool_input_hash}" if t.tool_input_hash else None,
        lambda key: f"{key.split('#')[0]} [{key.split('#')[1][:8]}]",
    )
    collect(
        "file edit",
        lambda t: t.edited_path,
        lambda key: key,
    )

    ranked = sorted(found, key=lambda r: (-r.clustered, -r.chars_in_cluster))
    return _deduplicate(ranked)


# A repeated call with byte-identical arguments is the stronger statement, so
# it wins over the file-edit view of the same turns.
KIND_PRIORITY = {"error": 0, "tool call": 1, "file edit": 2}


def _deduplicate(ranked: list[Repetition]) -> list[Repetition]:
    """Drop entries describing the same turns from a weaker angle.

    Repeating one tool call with identical arguments also repeats an edit to
    the same path, so both collectors fire and the output says the same thing
    twice.
    """
    kept: list[Repetition] = []
    for item in sorted(ranked, key=lambda r: KIND_PRIORITY.get(r.kind, 9)):
        duplicate = any(
            other.cluster_start == item.cluster_start
            and other.cluster_end == item.cluster_end
            and other.clustered == item.clustered
            for other in kept
        )
        if not duplicate:
            kept.append(item)
    return sorted(kept, key=lambda r: (-r.clustered, -r.chars_in_cluster))


def _since_last_clean(turns: list[TurnView]) -> tuple[int, int]:
    """Turns and characters since the last successful tool observation."""
    for offset, turn in enumerate(reversed(turns)):
        if turn.is_observation and not turn.is_error:
            tail = turns[len(turns) - offset :]
            return offset, sum(t.content_chars for t in tail)
    return len(turns), sum(t.content_chars for t in turns)


def analyse(
    session_id: str,
    turns: list[TurnView],
    scorer: Scorer | None = None,
    user_rejections: int = 0,
    top: int = 5,
) -> SessionReport:
    repetitions = _repetitions(turns)
    turns_since, chars_since = _since_last_clean(turns)

    trajectory: list[RiskPoint] = []
    scored_through = None
    if scorer is not None:
        usable = [c for c in scorer.cut_points if c <= len(turns)]
        for cut in usable:
            if cut < MIN_TURNS_TO_SCORE:
                continue
            report = scorer.score(session_id, turns[:cut])
            if report.failure_probability is not None:
                trajectory.append(RiskPoint(cut, report.failure_probability))
        scored_through = usable[-1] if usable else None

    return SessionReport(
        session_id=session_id,
        turns=len(turns),
        trajectory=tuple(trajectory),
        repetitions=tuple(repetitions[:top]),
        error_turns=sum(1 for t in turns if t.is_error),
        user_rejections=user_rejections,
        turns_since_clean=turns_since,
        chars_since_clean=chars_since,
        scored_through=scored_through,
    )


def render(report: SessionReport, base_rate: float | None = None) -> str:
    lines = [f"session {report.session_id}", f"turns: {report.turns:,}", ""]

    lines.append("MEASURED, exact, no model involved")
    lines.append(f"  agent errors            {report.error_turns}")
    lines.append(f"  user rejections         {report.user_rejections}  (not agent failures)")
    lines.append(f"  turns since a clean result  {report.turns_since_clean}")

    if report.repetitions:
        lines.append("")
        lines.append(f"  clustered repetition  (>={MIN_CLUSTERED}x within {WINDOW} turns)")
        for item in report.repetitions:
            spread = (
                f", {item.occurrences}x overall"
                if item.occurrences > item.clustered
                else ""
            )
            lines.append(
                f"    {item.clustered}x  {item.kind:<10} "
                f"turns {item.cluster_start}-{item.cluster_end}  "
                f"{item.chars_in_cluster:,} chars{spread}"
            )
            lines.append(f"         {item.label}")
    else:
        lines.append("")
        lines.append(
            f"  no clustered repetition, nothing recurred {MIN_CLUSTERED}+ times "
            f"within {WINDOW} turns"
        )

    if report.trajectory:
        lines.append("")
        lines.append("ESTIMATED, model did not clear its gate; context only")
        spark = _sparkline([p.probability for p in report.trajectory])
        lines.append(f"  risk over turns {report.trajectory[0].turn}-"
                     f"{report.trajectory[-1].turn}   {spark}  ({report.risk_direction})")
        for point in report.trajectory:
            bar = "█" * max(int(point.probability * 30), 1)
            lines.append(f"    turn {point.turn:>3}  {point.probability:>6.1%}  {bar}")
        if base_rate is not None:
            lines.append(f"  corpus base rate {base_rate:.1%}, compare against this, "
                         "not against zero")
        if report.scored_through and report.turns > report.scored_through:
            lines.append(
                f"  stops at turn {report.scored_through}: beyond the largest "
                "evaluated prefix, so no curve is drawn"
            )

    return "\n".join(lines)
