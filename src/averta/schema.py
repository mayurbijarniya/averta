"""Unified trajectory schema shared by every source adapter.

Adapters are the only code aware of source-specific field names. Everything
downstream reads these tables and nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb

SESSION_DDL = """
CREATE TABLE IF NOT EXISTS session (
    session_id   VARCHAR PRIMARY KEY,
    source       VARCHAR NOT NULL,
    agent        VARCHAR,
    repo         VARCHAR NOT NULL,
    instance_id  VARCHAR NOT NULL,
    run_id       VARCHAR,
    outcome      BOOLEAN NOT NULL,
    n_turns      INTEGER NOT NULL,
    n_steps      INTEGER NOT NULL,
    trajectory_hash VARCHAR NOT NULL,
    metadata     JSON
)
"""

TURN_DDL = """
CREATE TABLE IF NOT EXISTS turn (
    session_id      VARCHAR NOT NULL,
    turn_index      INTEGER NOT NULL,
    step_index      INTEGER,
    role            VARCHAR NOT NULL,
    tool_name       VARCHAR,
    tool_input      VARCHAR,
    tool_input_hash VARCHAR,
    tool_input_bad  BOOLEAN NOT NULL DEFAULT FALSE,
    n_tool_calls    INTEGER NOT NULL DEFAULT 0,
    content_chars   INTEGER NOT NULL DEFAULT 0,
    content_head    VARCHAR,
    is_error        BOOLEAN NOT NULL DEFAULT FALSE,
    error_signature VARCHAR,
    PRIMARY KEY (session_id, turn_index)
)
"""

INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS turn_session ON turn (session_id)",
    "CREATE INDEX IF NOT EXISTS session_repo ON session (repo)",
]

CONTENT_HEAD_CHARS = 400


@dataclass(slots=True)
class Turn:
    session_id: str
    turn_index: int
    role: str
    step_index: int | None = None
    tool_name: str | None = None
    tool_input: str | None = None
    tool_input_hash: str | None = None
    tool_input_bad: bool = False
    n_tool_calls: int = 0
    content_chars: int = 0
    content_head: str | None = None
    is_error: bool = False
    error_signature: str | None = None


@dataclass(slots=True)
class Session:
    session_id: str
    source: str
    repo: str
    instance_id: str
    outcome: bool
    n_turns: int
    n_steps: int
    trajectory_hash: str
    agent: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)


SESSION_COLUMNS = [
    "session_id",
    "source",
    "agent",
    "repo",
    "instance_id",
    "run_id",
    "outcome",
    "n_turns",
    "n_steps",
    "trajectory_hash",
    "metadata",
]

TURN_COLUMNS = [
    "session_id",
    "turn_index",
    "step_index",
    "role",
    "tool_name",
    "tool_input",
    "tool_input_hash",
    "tool_input_bad",
    "n_tool_calls",
    "content_chars",
    "content_head",
    "is_error",
    "error_signature",
]


def connect(path: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(path)
    conn.execute(SESSION_DDL)
    conn.execute(TURN_DDL)
    for statement in INDEX_DDL:
        conn.execute(statement)
    return conn


def write_sessions(conn: duckdb.DuckDBPyConnection, sessions: list[Session]) -> tuple[int, int]:
    if not sessions:
        return 0, 0

    import json

    session_rows = []
    turn_rows = []
    for session in sessions:
        record = asdict(session)
        record.pop("turns")
        record["metadata"] = json.dumps(record["metadata"])
        session_rows.append(tuple(record[column] for column in SESSION_COLUMNS))
        turn_rows.extend(
            tuple(asdict(turn)[column] for column in TURN_COLUMNS) for turn in session.turns
        )

    # Clear any prior turns for these sessions. Without this, replacing a
    # session with a shorter trajectory leaves the surplus turns orphaned
    # against a stale n_turns.
    conn.executemany(
        "DELETE FROM turn WHERE session_id = ?",
        [(session.session_id,) for session in sessions],
    )

    conn.executemany(
        f"INSERT OR REPLACE INTO session ({', '.join(SESSION_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(SESSION_COLUMNS))})",
        session_rows,
    )
    if turn_rows:
        conn.executemany(
            f"INSERT OR REPLACE INTO turn ({', '.join(TURN_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(TURN_COLUMNS))})",
            turn_rows,
        )
    return len(session_rows), len(turn_rows)
