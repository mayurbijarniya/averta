"""Builds the prefix feature matrix.

One row per (session, cut point). A cut point *t* is an absolute turn index,
never a fraction of the session — total length is unknown at inference time.
A session only contributes a row at *t* if it actually reached turn *t*, which
means the population changes as *t* grows. `resolve_rate` is reported per cut
so that shift stays visible.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb

from averta.features import FEATURE_NAMES, extract, to_views

CUT_POINTS = (3, 5, 10, 20, 40)

TURN_QUERY = """
SELECT session_id, turn_index, step_index, role, tool_name, tool_input,
       tool_input_hash, tool_input_bad, n_tool_calls, content_chars,
       is_error, error_signature
FROM turn
ORDER BY session_id, turn_index
"""

SESSION_QUERY = "SELECT session_id, source, repo, instance_id, run_id, outcome FROM session"


def prefix_table_ddl() -> str:
    columns = ",\n    ".join(f"{name} DOUBLE" for name in FEATURE_NAMES)
    return f"""
    CREATE TABLE IF NOT EXISTS prefix_features (
        session_id VARCHAR NOT NULL,
        cut_point  INTEGER NOT NULL,
        source     VARCHAR NOT NULL,
        repo       VARCHAR NOT NULL,
        instance_id VARCHAR NOT NULL,
        run_id     VARCHAR,
        outcome    BOOLEAN NOT NULL,
        {columns},
        PRIMARY KEY (session_id, cut_point)
    )
    """


def _grouped_turns(conn: duckdb.DuckDBPyConnection) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Stream turns grouped by session.

    Reads through an independent cursor. Issuing writes on the connection that
    is mid-stream resets its cursor state and truncates rows underneath the
    iteration.
    """
    cursor = conn.cursor().execute(TURN_QUERY)
    columns = [description[0] for description in cursor.description]

    current_id: str | None = None
    batch: list[dict[str, Any]] = []

    while rows := cursor.fetchmany(10_000):
        for row in rows:
            record = dict(zip(columns, row, strict=True))
            session_id = record["session_id"]
            if session_id != current_id:
                if current_id is not None:
                    yield current_id, batch
                current_id, batch = session_id, []
            batch.append(record)

    if current_id is not None:
        yield current_id, batch


def build(
    conn: duckdb.DuckDBPyConnection, cut_points: tuple[int, ...] = CUT_POINTS
) -> dict[str, Any]:
    conn.execute("DROP TABLE IF EXISTS prefix_features")
    conn.execute(prefix_table_ddl())

    meta = {
        row[0]: {
            "source": row[1],
            "repo": row[2],
            "instance_id": row[3],
            "run_id": row[4],
            "outcome": row[5],
        }
        for row in conn.execute(SESSION_QUERY).fetchall()
    }

    fixed = ["session_id", "cut_point", "source", "repo", "instance_id", "run_id", "outcome"]
    columns = fixed + list(FEATURE_NAMES)
    insert = (
        f"INSERT OR REPLACE INTO prefix_features ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(columns))})"
    )

    rows: list[tuple[Any, ...]] = []
    per_cut: dict[int, dict[str, int]] = {cut: {"rows": 0, "resolved": 0} for cut in cut_points}
    sessions_seen = 0

    for session_id, turns in _grouped_turns(conn):
        info = meta.get(session_id)
        if info is None:
            continue
        sessions_seen += 1
        views = to_views(turns)

        for cut in cut_points:
            if len(views) < cut:
                continue
            features = extract(views[:cut])
            rows.append(
                (
                    session_id,
                    cut,
                    info["source"],
                    info["repo"],
                    info["instance_id"],
                    info["run_id"],
                    info["outcome"],
                    *(features[name] for name in FEATURE_NAMES),
                )
            )
            per_cut[cut]["rows"] += 1
            per_cut[cut]["resolved"] += int(bool(info["outcome"]))

        if len(rows) >= 5_000:
            conn.executemany(insert, rows)
            rows.clear()

    if rows:
        conn.executemany(insert, rows)

    return {
        "sessions": sessions_seen,
        "features": len(FEATURE_NAMES),
        "cut_points": {
            cut: {
                "rows": stats["rows"],
                "resolved": stats["resolved"],
                "resolve_rate": round(stats["resolved"] / stats["rows"], 4)
                if stats["rows"]
                else None,
            }
            for cut, stats in per_cut.items()
        },
    }
