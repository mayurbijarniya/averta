"""Synthetic turn builders for tests."""

from __future__ import annotations

import json

from averta.features.view import TurnView
from averta.normalize import digest


def action(
    turn_index: int,
    tool: str = "str_replace_editor",
    *,
    command: str = "str_replace",
    path: str = "/workspace/a.py",
    step_index: int | None = None,
    chars: int = 0,
    malformed: bool = False,
) -> TurnView:
    payload = json.dumps({"command": command, "path": path})
    if malformed:
        payload = payload[:-4]
    return TurnView(
        turn_index=turn_index,
        role="assistant",
        step_index=turn_index if step_index is None else step_index,
        tool_name=tool,
        tool_input=payload,
        tool_input_hash=digest(payload),
        tool_input_bad=malformed,
        n_tool_calls=1,
        content_chars=chars,
        is_error=False,
        error_signature=None,
    )


def observation(
    turn_index: int,
    *,
    error: str | None = None,
    chars: int = 120,
) -> TurnView:
    return TurnView(
        turn_index=turn_index,
        role="tool",
        step_index=None,
        tool_name=None,
        tool_input=None,
        tool_input_hash=None,
        tool_input_bad=False,
        n_tool_calls=0,
        content_chars=chars,
        is_error=error is not None,
        error_signature=error,
    )


def bash(turn_index: int, command: str, **kwargs) -> TurnView:
    payload = json.dumps({"command": command})
    return action(turn_index, "execute_bash", **kwargs)._replace(
        tool_input=payload, tool_input_hash=digest(payload)
    )


def session(length: int = 30) -> list[TurnView]:
    """Alternating action/observation session with a recurring error."""
    turns: list[TurnView] = []
    for index in range(length):
        if index % 2 == 0:
            turns.append(action(index, path=f"/workspace/mod{index % 3}.py", chars=40))
        else:
            error = "ValueError: bad input" if index % 6 == 1 else None
            turns.append(observation(index, error=error))
    return turns
