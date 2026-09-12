"""Builders for synthetic sessions and transcripts.

Lives inside the package rather than under `tests/` so that both pytest and
type checkers resolve it from the single `src` import root, cross-importing
between test modules relies on pytest's path handling and nothing else agrees
with it.

Nothing here is used at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def observation(turn_index: int, *, error: str | None = None, chars: int = 120) -> TurnView:
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


def bash(turn_index: int, command: str, **kwargs: Any) -> TurnView:
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


def claude_assistant_record(
    text: str = "ok",
    tool: str | None = None,
    tool_input: dict[str, Any] | None = None,
    usage: dict[str, int] | None = None,
    cwd: str = "/home/me/proj",
) -> dict[str, Any]:
    """One `assistant` line as Claude Code writes it."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if tool:
        content.append({"type": "tool_use", "name": tool, "input": tool_input or {}})
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if usage:
        message["usage"] = usage
    return {"type": "assistant", "message": message, "cwd": cwd}


def claude_tool_result_record(
    body: str = "done", is_error: bool = False, cwd: str = "/home/me/proj"
) -> dict[str, Any]:
    """One `user` line carrying a tool result, as Claude Code writes it."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": body, "is_error": is_error}],
        },
        "cwd": cwd,
    }


def write_transcript(
    root: Path, records: list[dict[str, Any]], session_id: str = "abc123"
) -> Path:
    """Write records as a transcript under a Claude Code style directory.

    The project directory name starts with `-`, matching how Claude Code
    encodes an absolute cwd, which is also what makes these paths awkward for
    shell globbing.
    """
    project = root / "-some-project"
    project.mkdir(exist_ok=True)
    path = project / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path
