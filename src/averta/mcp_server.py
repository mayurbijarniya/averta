"""MCP server exposing Averta to a coding agent.

Runs locally over stdio. Nothing is transmitted anywhere: the agent asks about
its own session, the answer is computed from a 2.7 KB model on CPU, and the
agent's own tokens pay for the conversation.

Every response carries the accuracy caveats. The model did not clear its
pre-registered gate, and it is applied here to a different agent scaffold than
it was trained on, so an agent relaying these numbers must be able to relay the
limits with them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from averta.adapters import discover_transcripts, read_transcript
from averta.monitor import DEFAULT_MODEL_PATH, Scorer

mcp = MCPServer("averta")

GATE_CAVEAT = (
    "Averta did not meet its pre-registered decision gate: recall at a 5% "
    "false-positive budget reached 0.191 against a required 0.25 "
    "(AUROC 0.677). Treat this as a weak signal, not a verdict."
)

TRANSFER_CAVEAT = (
    "The model was trained on SWE-Gym OpenHands trajectories and is being "
    "applied to a Claude Code session. Cross-scaffold accuracy is unmeasured."
)


def _resolve(session: str | None) -> Path:
    paths = discover_transcripts()
    if not paths:
        raise ValueError("no transcripts found under ~/.claude/projects")
    if session is None:
        return paths[0]
    for path in paths:
        if path.stem.startswith(session):
            return path
    raise ValueError(f"no transcript matching {session!r}")


def _caveats() -> list[str]:
    return [GATE_CAVEAT, TRANSFER_CAVEAT]


@mcp.tool()
def get_session_risk(session: str | None = None, model_path: str | None = None) -> dict[str, Any]:
    """Estimate whether a coding session is heading toward failure.

    Args:
        session: session id prefix. Defaults to the most recently modified.
        model_path: override the model file. Defaults to artifacts/model.pkl.

    Returns the failure probability, the features driving it, and the accuracy
    caveats that must accompany any use of the number.
    """
    path = _resolve(session)
    transcript = read_transcript(path)
    scorer = Scorer.load(Path(model_path) if model_path else DEFAULT_MODEL_PATH)
    report = scorer.score(transcript.session_id, transcript.turns)

    return {
        "session_id": report.session_id,
        "project": transcript.repo,
        "turns_observed": report.turns,
        "scored_at_turn": report.scored_at_turn,
        "failure_probability": report.failure_probability,
        # Most sessions in the corpus fail, so the absolute probability means
        # little on its own. The comparison is what carries information.
        "corpus_base_rate": report.base_rate,
        "lift_over_base_rate": report.lift,
        "verdict": report.verdict,
        "tokens_used": transcript.total_tokens,
        "drivers": [
            {
                "feature": item.feature,
                "value": round(item.value, 4),
                "direction": "raises" if item.effect > 0 else "lowers",
            }
            for item in report.drivers
        ],
        "note": report.note,
        "caveats": _caveats(),
    }


@mcp.tool()
def get_repeated_failures(session: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List approaches already tried and failed in this session.

    Deterministic — no model involved. Errors are grouped by a normalized
    signature so the same failure recurring with different paths or line
    numbers collapses into one entry. Repeated tool calls are grouped by a hash
    of their exact arguments.
    """
    path = _resolve(session)
    transcript = read_transcript(path)

    errors: Counter[str] = Counter(
        turn.error_signature for turn in transcript.turns if turn.error_signature
    )

    calls: Counter[tuple[str | None, str | None]] = Counter(
        (turn.tool_name, turn.tool_input_hash)
        for turn in transcript.turns
        if turn.tool_input_hash
    )
    repeated = [(key, count) for key, count in calls.most_common() if count > 1]

    return {
        "session_id": transcript.session_id,
        "turns_observed": len(transcript),
        "recurring_errors": [
            {"signature": signature, "occurrences": count}
            for signature, count in errors.most_common(limit)
            if count > 1
        ],
        "reissued_calls": [
            # The argument hash distinguishes separate repeated calls to the
            # same tool, which would otherwise be indistinguishable here.
            {"tool": tool, "arguments_id": (call_hash or "")[:8], "occurrences": count}
            for (tool, call_hash), count in repeated[:limit]
        ],
        "distinct_errors": len(errors),
        "note": (
            "Recurrence is measured, not predicted; these counts are exact. "
            "Whether recurrence means the session is stuck is a judgement."
        ),
    }


@mcp.tool()
def should_i_restart(
    session: str | None = None, threshold: float = 0.75
) -> dict[str, Any]:
    """Recommend whether to restart, with the evidence behind it.

    The recommendation is a threshold applied to the risk estimate, not a
    separate model. Given the gate result it is advisory only, and the default
    threshold is deliberately high so the advice is rarely to stop.
    """
    risk = get_session_risk(session)
    probability = risk["failure_probability"]
    base_rate = risk["corpus_base_rate"]

    # A probability at or below the base rate carries no information, however
    # high it looks in absolute terms. Recommending a restart on that basis
    # would fire on almost every session.
    if probability is None:
        recommendation = "continue"
        reason = risk["note"] or "too few turns to assess"
    elif base_rate is not None and probability <= base_rate:
        recommendation = "continue"
        reason = (
            f"failure probability {probability:.0%} is at or below the "
            f"{base_rate:.0%} base rate — no evidence this session is unusual"
        )
    elif probability >= threshold:
        recommendation = "consider restarting"
        reason = (
            f"failure probability {probability:.0%} at or above the "
            f"{threshold:.0%} threshold, against a {base_rate:.0%} base rate"
            if base_rate is not None
            else f"failure probability {probability:.0%} at or above {threshold:.0%}"
        )
    else:
        recommendation = "continue"
        reason = (
            f"failure probability {probability:.0%} below the "
            f"{threshold:.0%} threshold"
        )

    return {
        "recommendation": recommendation,
        "reason": reason,
        "failure_probability": probability,
        "corpus_base_rate": base_rate,
        "verdict": risk["verdict"],
        "threshold": threshold,
        "tokens_used": risk["tokens_used"],
        "drivers": risk["drivers"],
        "caveats": _caveats()
        + [
            "At the measured operating point this catches under a fifth of "
            "failing sessions. A 'continue' recommendation is weak evidence "
            "that the session is fine."
        ],
    }


@mcp.tool()
def list_sessions(limit: int = 10) -> dict[str, Any]:
    """List local Claude Code sessions available to inspect."""
    sessions = []
    for path in discover_transcripts()[:limit]:
        transcript = read_transcript(path)
        sessions.append(
            {
                "session_id": transcript.session_id,
                "project": transcript.repo,
                "turns": len(transcript),
                "tokens": transcript.total_tokens,
            }
        )
    return {"sessions": sessions, "count": len(sessions)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
