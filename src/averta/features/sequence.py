"""Sequence-structure features over the action stream.

Attempt one used only aggregate counts over the prefix. Its strongest feature
by a wide margin was `n_repeated_calls`, a flat tally of calls reissued with
identical arguments, while error-based features contributed almost nothing.
That points at ordering as the missing ingredient: a count knows an action
recurred, not that the agent is cycling A-B-A-B, nor how far back it reached
to repeat itself.

These extractors read the same prefix and stay subject to the prefix-only
rule. They differ only in reading the *order* of actions rather than their
totals.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Sequence

from averta.features.view import TurnView

Prefix = Sequence[TurnView]

RECENT_ACTIONS = 6


def _symbols(prefix: Prefix) -> list[str]:
    """One opaque token per action, identifying tool plus exact arguments."""
    return [
        f"{turn.tool_name}#{turn.tool_input_hash}"
        for turn in prefix
        if turn.is_action and turn.tool_input_hash
    ]


def _ngram_repeat_max(symbols: Sequence[str], n: int) -> float:
    if len(symbols) < n:
        return 0.0
    grams = Counter(tuple(symbols[i : i + n]) for i in range(len(symbols) - n + 1))
    return float(max(grams.values()))


def action_bigram_repeat_max(prefix: Prefix) -> float:
    """Highest recurrence of any ordered pair of actions."""
    return _ngram_repeat_max(_symbols(prefix), 2)


def action_trigram_repeat_max(prefix: Prefix) -> float:
    return _ngram_repeat_max(_symbols(prefix), 3)


def longest_identical_run(prefix: Prefix) -> float:
    """Longest streak of consecutive byte-identical calls."""
    symbols = _symbols(prefix)
    if not symbols:
        return 0.0

    best = run = 1
    for previous, current in zip(symbols, symbols[1:], strict=False):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return float(best)


def alternation_count(prefix: Prefix) -> float:
    """Occurrences of A-B-A, the shortest form of oscillating between two states."""
    symbols = _symbols(prefix)
    return float(
        sum(
            1
            for i in range(len(symbols) - 2)
            if symbols[i] == symbols[i + 2] and symbols[i] != symbols[i + 1]
        )
    )


def distinct_action_ratio(prefix: Prefix) -> float:
    """Distinct actions over total. Falls as the agent recycles work."""
    symbols = _symbols(prefix)
    if not symbols:
        return 0.0
    return len(set(symbols)) / len(symbols)


def novelty_rate_recent(prefix: Prefix) -> float:
    """Share of the most recent actions never issued before them."""
    symbols = _symbols(prefix)
    if len(symbols) < 2:
        return 1.0

    recent = symbols[-RECENT_ACTIONS:]
    earlier = set(symbols[: -len(recent)])
    novel = 0
    for symbol in recent:
        if symbol not in earlier:
            novel += 1
        earlier.add(symbol)
    return novel / len(recent)


def mean_return_distance(prefix: Prefix) -> float:
    """Average gap, in actions, when an action repeats one seen earlier.

    Small values mean tight local cycling; large values mean the agent is
    circling back to something it abandoned a while ago.
    """
    symbols = _symbols(prefix)
    last_seen: dict[str, int] = {}
    gaps = []
    for index, symbol in enumerate(symbols):
        if symbol in last_seen:
            gaps.append(index - last_seen[symbol])
        last_seen[symbol] = index
    return float(sum(gaps) / len(gaps)) if gaps else 0.0


def action_compression_ratio(prefix: Prefix) -> float:
    """Compressed size over raw size for the action stream.

    A cheap global measure of redundancy: a repetitive sequence compresses
    further. Returns 1.0 when there is nothing to compress.

    Raw deflate is used rather than `zlib.compress`, whose ~11 byte header
    dominates short inputs, on local sessions that produced ratios of 1.85,
    which is meaningless for a quantity defined as compressed over raw. The
    result is still clamped, since even raw deflate adds a few bytes for
    incompressible input.
    """
    symbols = _symbols(prefix)
    if len(symbols) < 2:
        return 1.0

    # Map each distinct action to one byte so the ratio reflects sequence
    # structure rather than the length of the hash strings.
    alphabet: dict[str, int] = {}
    for symbol in symbols:
        alphabet.setdefault(symbol, len(alphabet) % 256)
    raw = bytes(alphabet[symbol] for symbol in symbols)

    compressor = zlib.compressobj(level=6, wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    return min(len(compressed) / len(raw), 1.0)


def edit_then_error_rate(prefix: Prefix) -> float:
    """Share of edits whose next observation was a failure."""
    edits = followed_by_error = 0
    for index, turn in enumerate(prefix):
        if not turn.edited_path:
            continue
        edits += 1
        for later in prefix[index + 1 :]:
            if later.is_observation:
                followed_by_error += int(later.is_error)
                break
    return followed_by_error / edits if edits else 0.0


def repeat_acceleration(prefix: Prefix) -> float:
    """Repeat density in the recent half against the earlier half.

    Above 1.0 means the agent is repeating itself more than it was, the shape
    a stalling session takes, as opposed to one that repeated early and then
    moved on.
    """
    symbols = _symbols(prefix)
    if len(symbols) < 4:
        return 0.0

    midpoint = len(symbols) // 2

    def density(chunk: Sequence[str]) -> float:
        if not chunk:
            return 0.0
        counts = Counter(chunk)
        return sum(n - 1 for n in counts.values()) / len(chunk)

    earlier = density(symbols[:midpoint])
    recent = density(symbols[midpoint:])
    if earlier == 0.0:
        return recent * 2.0 if recent > 0 else 0.0
    return recent / earlier


SEQUENCE_EXTRACTORS = {
    "action_bigram_repeat_max": action_bigram_repeat_max,
    "action_trigram_repeat_max": action_trigram_repeat_max,
    "longest_identical_run": longest_identical_run,
    "alternation_count": alternation_count,
    "distinct_action_ratio": distinct_action_ratio,
    "novelty_rate_recent": novelty_rate_recent,
    "mean_return_distance": mean_return_distance,
    "action_compression_ratio": action_compression_ratio,
    "edit_then_error_rate": edit_then_error_rate,
    "repeat_acceleration": repeat_acceleration,
}
