from __future__ import annotations

import json
from pathlib import Path

import typer

from averta.dataset import build as build_prefix_features
from averta.ingest import ADAPTERS
from averta.report import write_phase1_artifacts
from averta.schema import connect, write_sessions
from averta.thresholds import GATE_CUT_POINT
from averta.train import cross_validate, gate, render_table
from averta.train import load as load_dataset

app = typer.Typer(add_completion=False, help="Averta — failure prediction for coding agents.")

DEFAULT_DB = Path("data/averta.duckdb")
BATCH_SIZE = 250


@app.command()
def ingest(
    source: str = typer.Option("swegym", help=f"one of {', '.join(ADAPTERS)}"),
    db: Path = typer.Option(DEFAULT_DB, help="DuckDB file to write"),
    limit: int | None = typer.Option(None, help="stop after N records"),
) -> None:
    """Normalize a trajectory source into the unified schema."""
    if source not in ADAPTERS:
        raise typer.BadParameter(f"unknown source {source!r}")

    adapter = ADAPTERS[source]
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(str(db))

    batch = []
    read = skipped = sessions = turns = 0

    for record in adapter.load():
        if limit is not None and read >= limit:
            break
        read += 1

        session = adapter.convert(record)
        if session is None:
            skipped += 1
            continue

        batch.append(session)
        if len(batch) >= BATCH_SIZE:
            s, t = write_sessions(conn, batch)
            sessions, turns = sessions + s, turns + t
            batch.clear()
            typer.echo(f"  {read} read, {sessions} sessions, {turns} turns")

    if batch:
        s, t = write_sessions(conn, batch)
        sessions, turns = sessions + s, turns + t

    stored = conn.execute("SELECT count(*) FROM session WHERE source = ?", [source]).fetchone()[0]
    conn.close()

    typer.echo(f"\nread {read} records, skipped {skipped}")
    typer.echo(f"wrote {sessions} sessions and {turns} turns")
    typer.echo(f"{stored} distinct sessions stored for {source} ({sessions - stored} collapsed)")


@app.command()
def features(
    db: Path = typer.Option(DEFAULT_DB),
) -> None:
    """Build the prefix feature matrix from ingested sessions."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    conn = connect(str(db))
    try:
        summary = build_prefix_features(conn)
    finally:
        conn.close()

    typer.echo(f"sessions processed  {summary['sessions']}")
    typer.echo(f"features per row    {summary['features']}")
    typer.echo("\ncut  rows    resolved  resolve%")
    for cut, stats in summary["cut_points"].items():
        rate = f"{stats['resolve_rate']:.2%}" if stats["resolve_rate"] is not None else "n/a"
        typer.echo(f"{cut:>3}  {stats['rows']:>6}  {stats['resolved']:>8}  {rate:>8}")


@app.command()
def report(
    db: Path = typer.Option(DEFAULT_DB),
    out: Path = typer.Option(Path("artifacts/phase1"), help="directory for artifacts"),
) -> None:
    """Write corpus summary statistics and the turn distribution plot."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    summary = write_phase1_artifacts(str(db), out)

    typer.echo(f"sessions            {summary['sessions']}")
    typer.echo(f"resolved            {summary['resolved']} ({summary['resolve_rate']:.2%})")
    typer.echo(f"repos               {summary['distinct_repos']}")
    typer.echo(f"instances           {summary['distinct_instances']}")
    typer.echo(f"turn rows           {summary['turn_rows']}")
    typer.echo(f"error turns         {summary['error_turns']}")
    typer.echo(f"error signatures    {summary['distinct_error_signatures']}")
    typer.echo(f"malformed tool args {summary['malformed_tool_inputs']}")
    for row in summary["by_outcome"]:
        typer.echo(
            f"  resolved={row['resolved']}: n={row['sessions']}, "
            f"mean turns={row['mean_turns']}, mean steps={row['mean_steps']}"
        )
    typer.echo(f"\nwrote {out}/corpus_summary.json and {out}/turn_distribution.png")


@app.command()
def train(
    db: Path = typer.Option(DEFAULT_DB),
    cut: int = typer.Option(GATE_CUT_POINT, help="absolute turn index to score at"),
    folds: int = typer.Option(5, help="number of repo-grouped folds"),
    out: Path = typer.Option(Path("artifacts/phase3"), help="directory for results"),
) -> None:
    """Cross-validate every model at one cut point and apply the gate."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    data = load_dataset(str(db), cut)
    typer.echo(f"cut point {cut}: {len(data)} rows, failure rate {data.failure_rate:.2%}\n")

    results, diagnostics = cross_validate(data, folds)

    for warning in diagnostics["warnings"]:
        typer.echo(f"WARNING  {warning}")
    if diagnostics["warnings"]:
        typer.echo("")

    typer.echo(render_table(results, data.resolve_rate))

    best, verdict = gate(results)
    typer.echo(f"\nbest non-baseline model: {best.name}")
    typer.echo("")
    typer.echo(verdict.render())

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "cut_point": cut,
        "diagnostics": diagnostics,
        "results": [
            {
                key: value
                for key, value in vars(result).items()
                if key != "predictions"
            }
            for result in results
        ],
        "gate": {
            "model": best.name,
            "checks": verdict.checks,
            "passed": verdict.passed,
        },
    }
    with open(out / f"gate_cut{cut}.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    typer.echo(f"\nwrote {out / f'gate_cut{cut}.json'}")

    if not verdict.passed:
        raise typer.Exit(code=1)


CHECKS = {
    "sessions with zero turns": """
        SELECT s.session_id FROM session s
        LEFT JOIN turn t USING (session_id)
        GROUP BY s.session_id HAVING count(t.turn_index) = 0
    """,
    "turn count disagrees with session.n_turns": """
        SELECT s.session_id FROM session s
        JOIN turn t USING (session_id)
        GROUP BY s.session_id, s.n_turns HAVING count(*) <> s.n_turns
    """,
    "non-contiguous turn indices": """
        SELECT session_id FROM turn
        GROUP BY session_id
        HAVING max(turn_index) - min(turn_index) + 1 <> count(*) OR min(turn_index) <> 0
    """,
    "orphan turns": """
        SELECT DISTINCT t.session_id FROM turn t
        LEFT JOIN session s USING (session_id) WHERE s.session_id IS NULL
    """,
    "missing repo": "SELECT session_id FROM session WHERE repo IS NULL OR repo = ''",
    "missing instance id": "SELECT session_id FROM session WHERE instance_id IS NULL",
}


@app.command()
def validate(db: Path = typer.Option(DEFAULT_DB)) -> None:
    """Fail loudly on any structural problem in the store."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    conn = connect(str(db))
    failures = 0

    for label, query in CHECKS.items():
        offenders = conn.execute(query).fetchall()
        if offenders:
            failures += len(offenders)
            typer.echo(f"FAIL  {label}: {len(offenders)}")
            for (session_id,) in offenders[:3]:
                typer.echo(f"        {session_id}")
        else:
            typer.echo(f"ok    {label}")

    conn.close()

    if failures:
        typer.echo(f"\n{failures} problems found")
        raise typer.Exit(code=1)
    typer.echo("\nall checks passed")


@app.command()
def stats(db: Path = typer.Option(DEFAULT_DB)) -> None:
    """Summarize what is currently in the store."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    conn = connect(str(db))

    overview = conn.execute(
        """
        SELECT source,
               count(*) AS sessions,
               sum(outcome::INT) AS resolved,
               round(avg(outcome::INT) * 100, 2) AS resolve_pct,
               count(DISTINCT repo) AS repos,
               count(DISTINCT instance_id) AS instances,
               round(avg(n_turns), 1) AS mean_turns
        FROM session GROUP BY source ORDER BY source
        """
    ).fetchall()

    typer.echo("source    sessions  resolved  resolve%  repos  instances  mean turns")
    for row in overview:
        typer.echo(
            f"{row[0]:<9} {row[1]:>8}  {row[2]:>8}  {row[3]:>8}  {row[4]:>5}  "
            f"{row[5]:>9}  {row[6]:>10}"
        )

    turns_by_outcome = conn.execute(
        "SELECT outcome, count(*), round(avg(n_turns), 2) FROM session "
        "GROUP BY outcome ORDER BY outcome"
    ).fetchall()
    typer.echo("\nturns by outcome")
    for outcome, count, mean in turns_by_outcome:
        typer.echo(f"  resolved={outcome}: n={count}, mean turns={mean}")

    errors = conn.execute(
        """
        SELECT count(*) FILTER (WHERE is_error) AS error_turns,
               count(*) AS total_turns,
               count(DISTINCT error_signature) AS distinct_signatures
        FROM turn
        """
    ).fetchone()
    typer.echo(
        f"\nturns: {errors[1]} total, {errors[0]} flagged as errors, "
        f"{errors[2]} distinct error signatures"
    )

    tools = conn.execute(
        """
        SELECT tool_name, count(*) FROM turn
        WHERE tool_name IS NOT NULL GROUP BY tool_name ORDER BY 2 DESC
        """
    ).fetchall()
    typer.echo("\ntool vocabulary")
    for name, count in tools:
        typer.echo(f"  {name}: {count}")

    conn.close()


if __name__ == "__main__":
    app()
