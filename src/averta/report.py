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

# Drawn de-emphasised: these are reference lines, not candidate models.
BASELINE_MODELS = frozenset({"majority", "turn_index_only", "error_repeat_only"})


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


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)


def auroc_by_cut_plot(
    results: dict[int, dict[str, tuple[float, float, float]]], path: Path
) -> None:
    """AUROC against cut point, one line per model, with confidence bands.

    Populations differ between cut points — a session only appears at cut *t*
    if it reached turn *t* — so the lines are not a single model improving over
    time. The caption says so.
    """
    fig, axis = plt.subplots(figsize=(8, 4.5))
    cuts = sorted(results)

    for model in sorted({m for cut in results.values() for m in cut}):
        points = [results[c][model] for c in cuts if model in results[c]]
        xs = [c for c in cuts if model in results[c]]
        centres = [p[0] for p in points]
        lows = [p[1] for p in points]
        highs = [p[2] for p in points]

        is_baseline = model in BASELINE_MODELS
        (line,) = axis.plot(
            xs,
            centres,
            "--" if is_baseline else "-",
            linewidth=1.0 if is_baseline else 2.0,
            marker="." if is_baseline else "o",
            markersize=5 if is_baseline else 6,
            alpha=0.55 if is_baseline else 1.0,
            label=model,
        )
        if not is_baseline:
            axis.fill_between(xs, lows, highs, alpha=0.12, color=line.get_color())

    axis.axhline(0.5, color="#666", linewidth=0.8, linestyle=":")
    axis.set_xlabel("cut point (absolute turn index)")
    axis.set_ylabel("AUROC")
    axis.set_title("Failure prediction by how far into the session it is scored")
    axis.set_xticks(cuts)
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    _style(axis)

    fig.text(
        0.01,
        -0.04,
        "Bands are 95% intervals resampling repositories. Each cut point has a "
        "different population:\nonly sessions that reached that turn appear, so "
        "these are not one model tracked over time.",
        fontsize=7,
        color="#444",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_plot(
    raw: dict[str, list[float]],
    calibrated: dict[str, list[float]],
    path: Path,
    raw_brier: float | None = None,
    calibrated_brier: float | None = None,
) -> None:
    """Reliability before and after isotonic calibration.

    Both curves are shown because the raw one is the more instructive: class
    weighting is necessary for ranking and it is what puts the raw scores on
    the wrong scale. Showing only the fixed version would hide why the
    calibration step exists.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})

    axes[0].plot([0, 1], [0, 1], ":", color="#666", linewidth=0.9, label="perfect")
    raw_label = "raw (class-weighted)"
    if raw_brier is not None:
        raw_label += f" — Brier {raw_brier:.3f}"
    axes[0].plot(
        raw["predicted"], raw["observed"], "--s", color="#b03030",
        linewidth=1.4, markersize=4, alpha=0.85, label=raw_label,
    )

    cal_label = "isotonic calibrated"
    if calibrated_brier is not None:
        cal_label += f" — Brier {calibrated_brier:.3f}"
    axes[0].plot(
        calibrated["predicted"], calibrated["observed"], "-o",
        color=RESOLVED_COLOR, linewidth=2, label=cal_label,
    )

    axes[0].set_xlabel("predicted failure probability")
    axes[0].set_ylabel("observed failure rate")
    axes[0].set_title("Calibration")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    _style(axes[0])

    axes[1].bar(
        calibrated["predicted"], calibrated["counts"], width=0.08, color=UNRESOLVED_COLOR
    )
    axes[1].set_xlabel("calibrated failure probability")
    axes[1].set_ylabel("sessions")
    axes[1].set_title("Where predictions land")
    _style(axes[1])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def savings_tradeoff_plot(points: list[Any], path: Path) -> None:
    """Estimated savings against sessions wrongly terminated.

    Two y-axes because the quantities are not comparable and must not be
    summed: one is tokens that would not have been spent, the other is work
    that would have succeeded and was destroyed.
    """
    fig, axis = plt.subplots(figsize=(8, 4.5))

    fprs = [p.false_positive_rate for p in points]
    savings = [p.savings_rate * 100 for p in points]
    killed = [p.successes_terminated for p in points]

    axis.plot(fprs, savings, "-o", color=RESOLVED_COLOR, linewidth=2)
    axis.set_xlabel("false positive rate (sessions that would have resolved, flagged)")
    axis.set_ylabel("estimated tokens saved (% of total)", color=RESOLVED_COLOR)
    axis.tick_params(axis="y", labelcolor=RESOLVED_COLOR)
    axis.axvline(0.05, color="#b03030", linewidth=0.9, linestyle="--")
    axis.text(0.052, max(savings) * 0.9, "5% FPR budget", fontsize=8, color="#b03030")
    _style(axis)

    twin = axis.twinx()
    twin.plot(fprs, killed, "-s", color="#b03030", linewidth=1.2, markersize=4, alpha=0.8)
    twin.set_ylabel("successful sessions terminated", color="#b03030")
    twin.tick_params(axis="y", labelcolor="#b03030")
    twin.spines[["top"]].set_visible(False)

    axis.set_title("What intervention buys, and what it costs")
    fig.text(
        0.01,
        -0.04,
        "Tokens are estimated from content length; the corpus records none. "
        "The two axes are not\ncomparable quantities and are never summed.",
        fontsize=7,
        color="#444",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
