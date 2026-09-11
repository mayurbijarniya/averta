"""Corpus-level summaries written to artifacts/ for the README and phase notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESOLVED_COLOR = "#2f7d4f"
UNRESOLVED_COLOR = "#9aa0a6"


def corpus_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    sessions = conn.execute(
        """
        SELECT count(*), sum(outcome::INT), count(DISTINCT repo),
               count(DISTINCT instance_id), count(DISTINCT run_id)
        FROM session
        """
    ).fetchone()

    turn_quantiles = conn.execute(
        """
        SELECT min(n_turns), quantile_cont(n_turns, 0.25), median(n_turns),
               quantile_cont(n_turns, 0.75), max(n_turns), avg(n_turns)
        FROM session
        """
    ).fetchone()

    by_outcome = conn.execute(
        """
        SELECT outcome, count(*), avg(n_turns), avg(n_steps)
        FROM session GROUP BY outcome ORDER BY outcome
        """
    ).fetchall()

    turns = conn.execute(
        """
        SELECT count(*), count(*) FILTER (WHERE is_error),
               count(DISTINCT error_signature),
               count(*) FILTER (WHERE tool_input_bad)
        FROM turn
        """
    ).fetchone()

    tools = conn.execute(
        "SELECT tool_name, count(*) FROM turn WHERE tool_name IS NOT NULL "
        "GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()

    per_repo = conn.execute(
        """
        SELECT repo, count(*) AS sessions, sum(outcome::INT) AS resolved
        FROM session GROUP BY repo ORDER BY sessions DESC
        """
    ).fetchall()

    return {
        "sessions": sessions[0],
        "resolved": sessions[1],
        "resolve_rate": round(sessions[1] / sessions[0], 4) if sessions[0] else None,
        "distinct_repos": sessions[2],
        "distinct_instances": sessions[3],
        "distinct_runs": sessions[4],
        "turns": {
            "min": turn_quantiles[0],
            "p25": turn_quantiles[1],
            "median": turn_quantiles[2],
            "p75": turn_quantiles[3],
            "max": turn_quantiles[4],
            "mean": round(turn_quantiles[5], 2),
        },
        "by_outcome": [
            {
                "resolved": row[0],
                "sessions": row[1],
                "mean_turns": round(row[2], 2),
                "mean_steps": round(row[3], 2),
            }
            for row in by_outcome
        ],
        "turn_rows": turns[0],
        "error_turns": turns[1],
        "distinct_error_signatures": turns[2],
        "malformed_tool_inputs": turns[3],
        "tool_vocab": dict(tools),
        "per_repo": [
            {"repo": row[0], "sessions": row[1], "resolved": row[2]} for row in per_repo
        ],
    }


def turn_distribution_plot(conn: duckdb.DuckDBPyConnection, path: Path) -> None:
    rows = conn.execute("SELECT outcome, n_turns FROM session").fetchall()
    resolved = [n for outcome, n in rows if outcome]
    unresolved = [n for outcome, n in rows if not outcome]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})

    bins = range(0, max(r for _, r in rows) + 5, 5)
    axes[0].hist(
        [unresolved, resolved],
        bins=bins,
        density=True,
        color=[UNRESOLVED_COLOR, RESOLVED_COLOR],
        label=[f"unresolved (n={len(unresolved)})", f"resolved (n={len(resolved)})"],
    )
    axes[0].set_xlabel("turns per session")
    axes[0].set_ylabel("density")
    axes[0].set_title("Session length by outcome")
    axes[0].legend(frameon=False)

    parts = axes[1].boxplot(
        [unresolved, resolved], tick_labels=["unresolved", "resolved"], showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(parts["boxes"], [UNRESOLVED_COLOR, RESOLVED_COLOR], strict=True):
        patch.set_facecolor(color)
    axes[1].set_ylabel("turns")
    axes[1].set_title("Distribution overlap")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_phase1_artifacts(db: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db, read_only=True)
    try:
        summary = corpus_summary(conn)
        turn_distribution_plot(conn, out_dir / "turn_distribution.png")
    finally:
        conn.close()

    with open(out_dir / "corpus_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary
