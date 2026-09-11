"""Reads local Claude Code transcripts into the unified turn view.

Transcripts live at `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`,
one JSON object per line. They never leave the machine; nothing here uploads
or transmits.

The format differs from the training corpus in three ways that matter:

- Content is a list of typed blocks (`text`, `thinking`, `tool_use`,
  `tool_result`) rather than a string, and tool calls are blocks inside an
  assistant message rather than a separate `tool_calls` field.
- Tool results arrive as `tool_result` blocks inside *user* records, and carry
  an explicit `is_error` flag — more reliable than the text heuristics needed
  for the training data, which are kept only as a fallback.
- Real token counts are present in `message.usage`. The training corpus has
  none, so the model cannot use them as features, but they make token-saving
  estimates exact rather than inferred from character counts.

Many line types are editor bookkeeping (`file-history-snapshot`, `mode`,
`ai-title`, ...) and are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from averta.features.view import TurnView
from averta.normalize import digest, error_signature, is_valid_json, looks_like_error
from averta.schema import CONTENT_HEAD_CHARS

TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"

CONVERSATION_TYPES = frozenset({"user", "assistant", "system"})


@dataclass
class ClaudeCodeTranscript:
    session_id: str
    path: Path
    cwd: str | None
    git_branch: str | None
    turns: list[TurnView] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    malformed_lines: int = 0

    @property
    def repo(self) -> str:
        return Path(self.cwd).name if self.cwd else "unknown"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __len__(self) -> int:
        return len(self.turns)


def _blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _text_of(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for block in blocks:
        kind = block.get("type")
        if kind in ("text", "thinking"):
            parts.append(str(block.get(kind) or block.get("text") or ""))
        elif kind == "tool_result":
            parts.append(_tool_result_text(block))
    return "\n".join(part for part in parts if part)


def _tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )
    return ""


def read_transcript(path: Path) -> ClaudeCodeTranscript:
    transcript = ClaudeCodeTranscript(
        session_id=path.stem, path=path, cwd=None, git_branch=None
    )

    turn_index = 0
    step_index = 0

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                transcript.malformed_lines += 1
                continue

            if record.get("type") not in CONVERSATION_TYPES:
                continue

            transcript.cwd = transcript.cwd or record.get("cwd")
            transcript.git_branch = transcript.git_branch or record.get("gitBranch")

            message = record.get("message")
            blocks = _blocks(message)
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                transcript.input_tokens += int(usage.get("input_tokens") or 0)
                transcript.output_tokens += int(usage.get("output_tokens") or 0)

            for turn in _turns_from_record(record, blocks, turn_index, step_index):
                transcript.turns.append(turn)
                turn_index += 1
                if turn.step_index is not None:
                    step_index += 1

    return transcript


def _turns_from_record(
    record: dict[str, Any],
    blocks: list[dict[str, Any]],
    turn_index: int,
    step_index: int,
) -> Iterator[TurnView]:
    kind = record.get("type")
    text = _text_of(blocks)
    tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
    tool_results = [block for block in blocks if block.get("type") == "tool_result"]

    if kind == "assistant":
        tool_input = None
        tool_name = None
        if tool_uses:
            tool_name = tool_uses[0].get("name")
            tool_input = json.dumps(tool_uses[0].get("input"), sort_keys=True)

        yield TurnView(
            turn_index=turn_index,
            role="assistant",
            step_index=step_index,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_input_hash=digest(tool_input),
            tool_input_bad=bool(tool_input) and not is_valid_json(tool_input),
            n_tool_calls=len(tool_uses),
            content_chars=len(text),
            is_error=False,
            error_signature=None,
        )
        return

    if tool_results:
        for block in tool_results:
            body = _tool_result_text(block)
            # The explicit flag is authoritative; fall back to the text
            # heuristic only when it is absent.
            flagged = block.get("is_error")
            is_error = bool(flagged) if flagged is not None else looks_like_error(body)
            yield TurnView(
                turn_index=turn_index,
                role="tool",
                step_index=None,
                tool_name=None,
                tool_input=None,
                tool_input_hash=None,
                tool_input_bad=False,
                n_tool_calls=0,
                content_chars=len(body),
                is_error=is_error,
                error_signature=error_signature(body) if is_error else None,
            )
            turn_index += 1
        return

    yield TurnView(
        turn_index=turn_index,
        role="user" if kind == "user" else "system",
        step_index=None,
        tool_name=None,
        tool_input=None,
        tool_input_hash=None,
        tool_input_bad=False,
        n_tool_calls=0,
        content_chars=len(text),
        is_error=False,
        error_signature=None,
    )


def discover_transcripts(root: Path = TRANSCRIPT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def head(text: str) -> str | None:
    return text[:CONTENT_HEAD_CHARS] or None
