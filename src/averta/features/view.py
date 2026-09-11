"""Read-only turn projection handed to extractors."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, NamedTuple

OBSERVATION_ROLES = frozenset({"tool", "function"})

# Editing is expressed differently by each agent scaffold, and a feature that
# only understands one of them silently reports zero on the other. Measured
# against local Claude Code sessions, `n_edits` and `n_files_touched` came out
# at exactly 0.00 against corpus means of 4.11 and 1.78 for precisely this
# reason. Both vocabularies are recognised.
#
# OpenHands / SWE-Gym: one `str_replace_editor` tool, the operation in a
# `command` argument, target in `path`.
EDIT_COMMANDS = frozenset({"create", "str_replace", "insert", "write"})

# Claude Code: separate tools per operation, target in `file_path`.
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

PATH_KEYS = ("path", "file_path", "notebook_path", "filePath")

TEST_MARKERS = ("pytest", "unittest", "tox", "nosetests", " test", "test_")


class TurnView(NamedTuple):
    turn_index: int
    role: str
    step_index: int | None
    tool_name: str | None
    tool_input: str | None
    tool_input_hash: str | None
    tool_input_bad: bool
    n_tool_calls: int
    content_chars: int
    is_error: bool
    error_signature: str | None

    @property
    def is_observation(self) -> bool:
        return self.role in OBSERVATION_ROLES

    @property
    def is_action(self) -> bool:
        return self.role == "assistant" and self.n_tool_calls > 0

    def argument(self, key: str) -> Any:
        """Best-effort read of one tool argument. Malformed input yields None."""
        if not self.tool_input:
            return None
        try:
            payload = json.loads(self.tool_input)
        except (ValueError, TypeError):
            return None
        return payload.get(key) if isinstance(payload, dict) else None

    def first_path(self) -> str | None:
        for key in PATH_KEYS:
            value = self.argument(key)
            if isinstance(value, str) and value:
                return value
        return None

    @property
    def edited_path(self) -> str | None:
        """Target of a write, under either tool vocabulary."""
        if self.tool_name in EDIT_TOOLS:
            return self.first_path()
        if self.argument("command") in EDIT_COMMANDS:
            return self.first_path()
        return None

    @property
    def is_test_command(self) -> bool:
        command = self.argument("command")
        if not isinstance(command, str):
            return False
        lowered = command.lower()
        return any(marker in lowered for marker in TEST_MARKERS)


def to_views(rows: Sequence[dict[str, Any]]) -> list[TurnView]:
    return [
        TurnView(
            turn_index=row["turn_index"],
            role=row["role"],
            step_index=row.get("step_index"),
            tool_name=row.get("tool_name"),
            tool_input=row.get("tool_input"),
            tool_input_hash=row.get("tool_input_hash"),
            tool_input_bad=bool(row.get("tool_input_bad")),
            n_tool_calls=row.get("n_tool_calls") or 0,
            content_chars=row.get("content_chars") or 0,
            is_error=bool(row.get("is_error")),
            error_signature=row.get("error_signature"),
        )
        for row in rows
    ]
