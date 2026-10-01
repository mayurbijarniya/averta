"""Reads local Claude Code transcripts into the unified turn view.

Transcripts live at `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`,
one JSON object per line. `select_transcript` decides which one a command or
an agent means. They never leave the machine; nothing here uploads
or transmits.

The format differs from the training corpus in four ways that matter:

- Content is a list of typed blocks (`text`, `thinking`, `tool_use`,
  `tool_result`) rather than a string, and tool calls are blocks inside an
  assistant message rather than a separate `tool_calls` field.
- One model response is written as several `assistant` lines, one per content
  block, each repeating the response's `message.id` and its full `usage`.
  With parallel tool calls the lines are not even adjacent: the first call's
  result is logged before the second call. Lines are therefore merged by
  `message.id` into one assistant turn, matching the corpus, where one message
  is one turn however many tool calls it carries, and usage is counted once
  per response. Treating each line as a turn double-counted tokens and
  inflated every turn index.
- Tool results arrive as `tool_result` blocks inside *user* records, and carry
  an explicit `is_error` flag, more reliable than the text heuristics needed
  for the training data, which are kept only as a fallback.
- Real token counts are present in `message.usage`. The training corpus has
  none, so the model cannot use them as features, but they make token-saving
  estimates exact rather than inferred from character counts.

Many line types are editor bookkeeping (`file-history-snapshot`, `mode`,
`ai-title`, ...) and are skipped.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from averta.features.view import TurnView
from averta.normalize import (
    digest,
    error_signature,
    is_user_rejection,
    is_valid_json,
    looks_like_error,
)

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
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    malformed_lines: int = 0
    user_rejections: int = 0
    # Last cost snapshot written by the editor. Lags the live session; a lower
    # bound rather than the final figure.
    cost_usd_snapshot: float | None = None

    @property
    def repo(self) -> str:
        return Path(self.cwd).name if self.cwd else "unknown"

    @property
    def total_tokens(self) -> int:
        """All input and output, cached or not.

        `usage.input_tokens` counts only uncached input. Under prompt caching
        that is a rounding error, one record showed `input_tokens: 2` beside
        `cache_read_input_tokens: 18,641`. Summing the plain field alone
        undercounted a real session's input by three orders of magnitude.
        """
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

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


@dataclass
class _Entry:
    """One conversational message, possibly assembled from several lines."""

    kind: str
    blocks: list[dict[str, Any]]


def _response_id(record: dict[str, Any]) -> str | None:
    message = record.get("message")
    if record.get("type") != "assistant" or not isinstance(message, dict):
        return None
    response_id = message.get("id")
    return response_id if isinstance(response_id, str) and response_id else None


def read_transcript(path: Path) -> ClaudeCodeTranscript:
    transcript = ClaudeCodeTranscript(
        session_id=path.stem, path=path, cwd=None, git_branch=None
    )

    entries: list[_Entry] = []
    responses: dict[str, _Entry] = {}
    # Keyed by response id, so a response split over several lines counts
    # once. Lines without an id are each their own response.
    usage_by_response: dict[str | int, dict[str, Any]] = {}

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                transcript.malformed_lines += 1
                continue
            if not isinstance(record, dict):
                transcript.malformed_lines += 1
                continue

            # Cost is recorded directly by the editor, which is better than
            # deriving it, that would need current per-model pricing and the
            # cache-read discount. But the record is a periodic snapshot, not
            # a running total: on a multi-hour session the last one carried
            # `totalDuration` of 876s. Treat it as a lower bound.
            if record.get("type") == "cost-state":
                cost = record.get("totalCostUSD")
                if isinstance(cost, int | float):
                    transcript.cost_usd_snapshot = float(cost)
                continue

            if record.get("type") not in CONVERSATION_TYPES:
                continue

            transcript.cwd = transcript.cwd or record.get("cwd")
            transcript.git_branch = transcript.git_branch or record.get("gitBranch")

            message = record.get("message")
            blocks = _blocks(message)
            response_id = _response_id(record)

            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                usage_by_response[response_id or line_number] = usage

            # A continuation line extends the turn where the response began,
            # so its tool calls stay ahead of the results that answer them.
            if response_id is not None and response_id in responses:
                responses[response_id].blocks.extend(blocks)
                continue

            entry = _Entry(kind=record["type"], blocks=list(blocks))
            entries.append(entry)
            if response_id is not None:
                responses[response_id] = entry

    for usage in usage_by_response.values():
        transcript.input_tokens += int(usage.get("input_tokens") or 0)
        transcript.output_tokens += int(usage.get("output_tokens") or 0)
        transcript.cache_read_tokens += int(usage.get("cache_read_input_tokens") or 0)
        transcript.cache_write_tokens += int(usage.get("cache_creation_input_tokens") or 0)

    turn_index = 0
    step_index = 0
    for entry in entries:
        for turn, rejected in _turns_from_entry(entry, turn_index, step_index):
            transcript.turns.append(turn)
            transcript.user_rejections += int(rejected)
            turn_index += 1
            if turn.step_index is not None:
                step_index += 1

    return transcript


def _turns_from_entry(
    entry: _Entry, turn_index: int, step_index: int
) -> Iterator[tuple[TurnView, bool]]:
    """Yields each turn with a flag for whether it was a user rejection."""
    kind = entry.kind
    blocks = entry.blocks
    text = _text_of(blocks)
    tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
    tool_results = [block for block in blocks if block.get("type") == "tool_result"]

    if kind == "assistant":
        tool_input = None
        tool_name = None
        if tool_uses:
            tool_name = tool_uses[0].get("name")
            tool_input = json.dumps(tool_uses[0].get("input"), sort_keys=True)

        yield (
            TurnView(
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
            ),
            False,
        )
        return

    if tool_results:
        for block in tool_results:
            body = _tool_result_text(block)
            # The explicit flag is authoritative; fall back to the text
            # heuristic only when it is absent.
            flagged = block.get("is_error")
            explicit = flagged is not None
            failed = bool(flagged) if explicit else looks_like_error(body)

            # A declined tool call is flagged as an error by the editor, but
            # it is a human decision, not the agent failing. Counting it would
            # make a closely supervised session look like a struggling one.
            rejected = failed and is_user_rejection(body)
            is_error = failed and not rejected

            yield (
                TurnView(
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
                    error_signature=(
                        error_signature(body, force=explicit) if is_error else None
                    ),
                ),
                rejected,
            )
            turn_index += 1
        return

    yield (
        TurnView(
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
        ),
        False,
    )


def transcript_root() -> Path:
    """Where Claude Code keeps transcripts, resolved at call time.

    Honours `CLAUDE_CONFIG_DIR`, which Claude Code uses to relocate its whole
    config directory, and otherwise follows `$HOME`, so a test or a sandbox
    can point it elsewhere without patching anything.
    """
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config).expanduser() if config else Path.home() / ".claude"
    return base / "projects"


def discover_transcripts(root: Path | None = None) -> list[Path]:
    """Every transcript under the root, most recently written first."""
    root = transcript_root() if root is None else root
    if not root.exists():
        return []
    return sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def transcript_cwd(path: Path) -> str | None:
    """The directory the session was started in.

    Read from the first record that carries `cwd`, without parsing the rest.
    The project folder name is a lossy encoding of the same path, every
    non-alphanumeric character becomes `-`, so it cannot be decoded reliably
    and is not used.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"cwd"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                cwd = record.get("cwd") if isinstance(record, dict) else None
                if isinstance(cwd, str) and cwd:
                    return cwd
    except OSError:
        return None
    return None


def _identity(path: str | Path) -> tuple[int, int] | str:
    """A key under which two spellings of the same directory compare equal.

    Device and inode when the directory exists, which absorbs symlinks and
    case-insensitive filesystems: on macOS `~/Downloads/x` and `~/downloads/x`
    are one directory but two strings. A deleted directory falls back to its
    normalized path.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))
    return (stat.st_dev, stat.st_ino)


def _lineage(here: Path) -> list[tuple[int, int] | str]:
    """Identities of `here` and each of its parents, nearest first."""
    resolved = Path(os.path.abspath(here))
    return [_identity(p) for p in (resolved, *resolved.parents)]


def project_transcripts(here: Path, paths: list[Path]) -> list[tuple[Path, int]]:
    """Transcripts belonging to the project at `here`, with their distance.

    Distance 0 means the session was started in `here` itself; 1 means in its
    parent, and so on, which covers running from a subdirectory of the
    project. Sessions started *below* `here` are not included: run from a home
    directory, that would claim every session on the machine.

    Only the nearest level that has any session is returned, most recent
    first. An exact match always beats a fresher session that merely started
    in an enclosing directory, and a workspace folder holding many projects
    (`~/code`) does not leak its sessions into each project's listing.
    """
    # The filesystem root and the home directory enclose nearly everything, so
    # as ancestors they identify nothing. Either still matches exactly.
    lineage = _lineage(here)
    generic = {_identity(Path.home()), _identity(Path(os.path.abspath(here)).anchor)}
    position = {lineage[0]: 0}
    for depth, identity in enumerate(lineage[1:], start=1):
        if identity not in generic:
            position.setdefault(identity, depth)

    matched = []
    for order, path in enumerate(paths):
        cwd = transcript_cwd(path)
        if cwd is None:
            continue
        depth = position.get(_identity(cwd))
        if depth is not None:
            matched.append((depth, order, path))
    if not matched:
        return []
    nearest = min(depth for depth, _, _ in matched)
    return [(path, depth) for depth, _, path in sorted(matched) if depth == nearest]


SESSION_ENV = "AVERTA_SESSION"
PROJECT_ENV = "AVERTA_PROJECT"

# A sibling written this recently is probably a second live session.
CONCURRENT_WINDOW_SECONDS = 300


@dataclass(frozen=True)
class Selection:
    """Which transcript was chosen and why, so the choice can be checked."""

    path: Path
    selected_by: str
    project_dir: str
    session_dir: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def session_id(self) -> str:
        return self.path.stem

    def describe(self) -> str:
        reasons = {
            "explicit": "matched the session id given",
            "env": f"pinned by ${SESSION_ENV}",
            "project": f"most recent session started in {self.project_dir}",
            "parent": f"most recent session started in {self.session_dir}, "
            f"which encloses {self.project_dir}",
            "fallback": "most recent session in any project",
        }
        return reasons[self.selected_by]

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "transcript_path": str(self.path),
            "session_dir": self.session_dir,
            "selected_by": self.selected_by,
            "selected_because": self.describe(),
            "project_dir": self.project_dir,
            "warnings": list(self.warnings),
        }


class SelectionError(ValueError):
    pass


def current_project() -> Path:
    """`$AVERTA_PROJECT` if set, otherwise the working directory."""
    override = os.environ.get(PROJECT_ENV)
    return Path(override).expanduser() if override else Path.cwd()


def _match_id(session: str, paths: list[Path]) -> Path:
    exact = [p for p in paths if p.stem == session]
    if exact:
        return exact[0]
    matches = [p for p in paths if p.stem.startswith(session)]
    if not matches:
        raise SelectionError(f"no transcript matching {session!r}")
    if len(matches) > 1:
        listed = ", ".join(p.stem[:12] for p in matches[:5])
        raise SelectionError(
            f"{session!r} matches {len(matches)} transcripts ({listed}); give more of the id"
        )
    return matches[0]


def select_transcript(
    session: str | None = None,
    here: Path | None = None,
    paths: list[Path] | None = None,
) -> Selection:
    """Choose the transcript a command or an agent is asking about.

    In order: an explicit session id, then `$AVERTA_SESSION`, then the most
    recent session started in the project at `here` (or `$AVERTA_PROJECT`,
    or the working directory), then the most recent session anywhere, which
    is flagged because it is probably not the one meant.

    Session ids are matched across every project: they are UUIDs, and
    restricting them to the current project would make an explicit request
    fail for no reason.
    """
    here = current_project() if here is None else here
    project_dir = os.path.abspath(here)

    paths = discover_transcripts() if paths is None else paths
    if not paths:
        raise SelectionError(f"no transcripts found under {transcript_root()}")

    def chosen(path: Path, selected_by: str, warnings: list[str]) -> Selection:
        return Selection(path, selected_by, project_dir, transcript_cwd(path), tuple(warnings))

    if session:
        return chosen(_match_id(session, paths), "explicit", [])

    pinned = os.environ.get(SESSION_ENV)
    if pinned:
        return chosen(_match_id(pinned, paths), "env", [])

    in_project = project_transcripts(here, paths)
    if in_project:
        path, depth = in_project[0]
        warnings = []
        siblings = [p for p, d in in_project[1:] if d == depth]
        if siblings:
            gap = path.stat().st_mtime - siblings[0].stat().st_mtime
            if gap < CONCURRENT_WINDOW_SECONDS:
                warnings.append(
                    f"another session here ({siblings[0].stem[:8]}) was written "
                    f"{gap:.0f}s before this one; if two sessions are open, pass the "
                    f"session id or set ${SESSION_ENV} to be sure which one is read"
                )
        return chosen(path, "project" if depth == 0 else "parent", warnings)

    return chosen(
        paths[0],
        "fallback",
        [
            f"no session was started in {project_dir}; this is the most recent "
            "session from any project, which may not be the one you meant"
        ],
    )
