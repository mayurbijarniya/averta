"""Feature functions. Each takes a prefix and returns one float.

Naming rule: a feature is named for what it measures, not what it is hoped to
indicate. `max_error_repeat` counts a repeated error signature; it does not
claim the agent is "stuck".
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from averta.features.view import TurnView

Prefix = Sequence[TurnView]

RECENT_WINDOW = 5


def _observations(prefix: Prefix) -> list[TurnView]:
    return [turn for turn in prefix if turn.is_observation]


def _actions(prefix: Prefix) -> list[TurnView]:
    return [turn for turn in prefix if turn.is_action]


def turns_seen(prefix: Prefix) -> float:
    return float(len(prefix))


def n_steps(prefix: Prefix) -> float:
    return float(sum(1 for turn in prefix if turn.step_index is not None))


def n_tool_calls(prefix: Prefix) -> float:
    return float(sum(turn.n_tool_calls for turn in prefix))


def n_distinct_tools(prefix: Prefix) -> float:
    return float(len({turn.tool_name for turn in prefix if turn.tool_name}))


def tool_entropy(prefix: Prefix) -> float:
    """Shannon entropy over tool usage. Low entropy means one tool dominates."""
    counts = Counter(turn.tool_name for turn in prefix if turn.tool_name)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def n_errors(prefix: Prefix) -> float:
    return float(sum(1 for turn in prefix if turn.is_error))


def error_rate(prefix: Prefix) -> float:
    observations = _observations(prefix)
    if not observations:
        return 0.0
    return sum(1 for turn in observations if turn.is_error) / len(observations)


def n_distinct_errors(prefix: Prefix) -> float:
    return float(len({turn.error_signature for turn in prefix if turn.error_signature}))


def max_error_repeat(prefix: Prefix) -> float:
    """Highest number of times a single error signature has recurred."""
    counts = Counter(turn.error_signature for turn in prefix if turn.error_signature)
    return float(max(counts.values())) if counts else 0.0


def turns_since_error(prefix: Prefix) -> float:
    for offset, turn in enumerate(reversed(prefix)):
        if turn.is_error:
            return float(offset)
    return float(len(prefix))


def turns_since_clean_observation(prefix: Prefix) -> float:
    """Turns elapsed with no successful tool result. A stall proxy."""
    for offset, turn in enumerate(reversed(prefix)):
        if turn.is_observation and not turn.is_error:
            return float(offset)
    return float(len(prefix))


def n_repeated_calls(prefix: Prefix) -> float:
    """Tool calls issued with byte-identical arguments to an earlier call."""
    counts = Counter(
        (turn.tool_name, turn.tool_input_hash) for turn in prefix if turn.tool_input_hash
    )
    return float(sum(n - 1 for n in counts.values() if n > 1))


def max_call_repeat(prefix: Prefix) -> float:
    counts = Counter(
        (turn.tool_name, turn.tool_input_hash) for turn in prefix if turn.tool_input_hash
    )
    return float(max(counts.values())) if counts else 0.0


def recent_call_repeat_rate(prefix: Prefix) -> float:
    """Share of the last few calls that repeat an argument seen earlier."""
    hashes = [turn.tool_input_hash for turn in prefix if turn.tool_input_hash]
    if len(hashes) < 2:
        return 0.0
    recent = hashes[-RECENT_WINDOW:]
    earlier = set(hashes[: -len(recent)])
    if not earlier:
        return 0.0
    return sum(1 for h in recent if h in earlier) / len(recent)


def n_files_touched(prefix: Prefix) -> float:
    return float(len({turn.edited_path for turn in prefix if turn.edited_path}))


def max_file_edit_repeat(prefix: Prefix) -> float:
    """Most edits applied to any single file. High values indicate churn."""
    counts = Counter(turn.edited_path for turn in prefix if turn.edited_path)
    return float(max(counts.values())) if counts else 0.0


def n_edits(prefix: Prefix) -> float:
    return float(sum(1 for turn in prefix if turn.edited_path))


def n_test_commands(prefix: Prefix) -> float:
    return float(sum(1 for turn in prefix if turn.is_test_command))


def n_malformed_tool_inputs(prefix: Prefix) -> float:
    return float(sum(1 for turn in prefix if turn.tool_input_bad))


def chars_total(prefix: Prefix) -> float:
    return float(sum(turn.content_chars for turn in prefix))


def chars_recent(prefix: Prefix) -> float:
    return float(sum(turn.content_chars for turn in prefix[-RECENT_WINDOW:]))


def chars_growth_ratio(prefix: Prefix) -> float:
    """Recent output volume against the session's own average.

    Above 1.0 means the agent is producing more per turn than it has been.
    """
    if not prefix:
        return 0.0
    mean_per_turn = chars_total(prefix) / len(prefix)
    if mean_per_turn == 0:
        return 0.0
    window = prefix[-RECENT_WINDOW:]
    return (chars_recent(prefix) / len(window)) / mean_per_turn


def has_finished(prefix: Prefix) -> float:
    return float(any(turn.tool_name == "finish" for turn in prefix))


EXTRACTORS: dict[str, Callable[[Prefix], float]] = {
    "turns_seen": turns_seen,
    "n_steps": n_steps,
    "n_tool_calls": n_tool_calls,
    "n_distinct_tools": n_distinct_tools,
    "tool_entropy": tool_entropy,
    "n_errors": n_errors,
    "error_rate": error_rate,
    "n_distinct_errors": n_distinct_errors,
    "max_error_repeat": max_error_repeat,
    "turns_since_error": turns_since_error,
    "turns_since_clean_observation": turns_since_clean_observation,
    "n_repeated_calls": n_repeated_calls,
    "max_call_repeat": max_call_repeat,
    "recent_call_repeat_rate": recent_call_repeat_rate,
    "n_files_touched": n_files_touched,
    "max_file_edit_repeat": max_file_edit_repeat,
    "n_edits": n_edits,
    "n_test_commands": n_test_commands,
    "n_malformed_tool_inputs": n_malformed_tool_inputs,
    "chars_total": chars_total,
    "chars_recent": chars_recent,
    "chars_growth_ratio": chars_growth_ratio,
    "has_finished": has_finished,
}

FEATURE_NAMES = tuple(EXTRACTORS)


def extract(prefix: Prefix) -> dict[str, float]:
    return {name: fn(prefix) for name, fn in EXTRACTORS.items()}
