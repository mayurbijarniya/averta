"""Source-agnostic normalization: repository keys and error identity.

Error signatures exist so that the same failure recurring across turns is
recognizable. Paths, line numbers, addresses and digits are stripped, because
a traceback that differs only in a temp directory name is the same error.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

INSTANCE_PATTERN = re.compile(r"^(?P<owner>[^_]+(?:_[^_]+)*)__(?P<repo>.+?)-(?P<suffix>[^-]+)$")

TRACEBACK_MARKER = "Traceback (most recent call last)"

EXCEPTION_LINE = re.compile(r"^\s*(?P<name>[A-Z]\w*(?:Error|Exception|Warning))\b:?\s*(?P<msg>.*)$")

ERROR_MARKERS = (
    TRACEBACK_MARKER,
    "command not found",
    "no such file or directory",
    "permission denied",
    "syntaxerror",
    "modulenotfounderror",
    "importerror",
    "assertionerror",
    "did not appear verbatim",
    "failed to",
    "error:",
    "fatal:",
)

EXIT_CODE = re.compile(r"exit code[:=]?\s*(\d+)", re.IGNORECASE)

UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

SCRUB = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    (UUID_PATTERN, "<uuid>"),
    (re.compile(r"(/[\w.\-]+){2,}"), "<path>"),
    (re.compile(r"line \d+"), "line <n>"),
    (re.compile(r"\d+"), "<n>"),
    (re.compile(r"\s+"), " "),
]

MAX_SIGNATURE_CHARS = 180


def digest(text: str | None) -> str | None:
    """Stable short hash of a tool input, for detecting reissued calls."""
    if text is None:
        return None
    collapsed = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha1(collapsed.encode("utf-8", "replace")).hexdigest()[:16]


def trajectory_hash(parts: Iterable[str]) -> str:
    """Content hash of a turn sequence.

    Identifies a trajectory independently of the identifiers attached to it,
    so records that merely share an instance and run id are not mistaken for
    duplicates of each other.
    """
    accumulator = hashlib.sha1()
    for part in parts:
        accumulator.update(part.encode("utf-8", "replace"))
        accumulator.update(b"\x1f")
    return accumulator.hexdigest()


def is_valid_json(text: str | None) -> bool:
    if not text:
        return False
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def parse_instance_id(instance_id: str) -> tuple[str | None, str | None]:
    """Split 'getmoto__moto-5321' into ('getmoto/moto', '5321')."""
    match = INSTANCE_PATTERN.match(instance_id)
    if not match:
        return None, None
    return f"{match['owner']}/{match['repo']}", match["suffix"]


def scrub(text: str) -> str:
    for pattern, replacement in SCRUB:
        text = pattern.sub(replacement, text)
    return text.strip()


HEAD_WINDOW = 300


def looks_like_error(content: str) -> bool:
    """Detect a failed observation.

    Markers are only trusted near the start of the output. Tool results that
    return file contents routinely contain the word "error" in source code,
    and treating those as failures inflates the repeat-error signal.
    """
    if not content:
        return False

    if TRACEBACK_MARKER in content:
        return True

    head = content[:HEAD_WINDOW].lower()
    if any(marker in head for marker in ERROR_MARKERS):
        return True

    match = EXIT_CODE.search(content[:HEAD_WINDOW])
    return bool(match and match.group(1) != "0")


def error_signature(content: str) -> str | None:
    """Reduce an error observation to a stable identity string."""
    if not content or not looks_like_error(content):
        return None

    lines = [line for line in content.splitlines() if line.strip()]

    if TRACEBACK_MARKER in content:
        for line in reversed(lines):
            if match := EXCEPTION_LINE.match(line):
                return scrub(f"{match['name']}: {match['msg']}")[:MAX_SIGNATURE_CHARS]

    for line in lines:
        if match := EXCEPTION_LINE.match(line):
            return scrub(f"{match['name']}: {match['msg']}")[:MAX_SIGNATURE_CHARS]

    lowered = content.lower()
    for marker in ERROR_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            return scrub(content[index : index + MAX_SIGNATURE_CHARS])[:MAX_SIGNATURE_CHARS]

    if match := EXIT_CODE.search(content):
        return f"exit code {match.group(1)}"

    return None
