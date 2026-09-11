from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from averta.adapters import discover_transcripts, read_transcript
from averta.analyze import (
    calibration,
    inference_cost,
    out_of_fold_predictions,
    permutation_importance,
    render_cost,
    render_importance,
)
from averta.dataset import CUT_POINTS
from averta.dataset import build as build_prefix_features
from averta.drift import compare
from averta.features import FEATURE_NAMES, extract
from averta.ingest import ADAPTERS
from averta.metrics import brier_score
from averta.monitor import DEFAULT_MODEL_PATH, Calibrator, Scorer, fit_and_save
from averta.report import (
    auroc_by_cut_plot,
    calibration_plot,
    savings_tradeoff_plot,
    write_phase1_artifacts,
)
from averta.savings import load_token_estimates, simulate
from averta.savings import render as render_savings
from averta.schema import connect, write_sessions
from averta.site import build as build_site
from averta.thresholds import GATE_CUT_POINT
from averta.train import cross_validate, gate, load_pooled, render_table
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


@app.command()
def diagnose(
    db: Path = typer.Option(DEFAULT_DB),
    cut: int = typer.Option(GATE_CUT_POINT),
    model: str = typer.Option("logistic", help="model to analyse in depth"),
    out: Path = typer.Option(Path("artifacts/phase4")),
) -> None:
    """Feature importance, calibration, and CPU inference cost."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    data = load_dataset(str(db), cut)
    typer.echo(f"cut point {cut}: {len(data)} rows, failure rate {data.failure_rate:.2%}\n")

    typer.echo(f"permutation importance ({model}, repo-grouped folds)")
    importance = permutation_importance(data, model)
    typer.echo(render_importance(importance))

    typer.echo("\nCPU inference cost")
    profiles = [
        inference_cost(data, name)
        for name in ("logistic", "random_forest", "hist_gradient_boosting", "xgboost")
    ]
    typer.echo(render_cost(profiles))

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "cut_point": cut,
        "model": model,
        "importance": [vars(item) for item in importance],
        "cost": [vars(profile) for profile in profiles],
    }
    with open(out / f"diagnostics_cut{cut}.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    typer.echo(f"\nwrote {out / f'diagnostics_cut{cut}.json'}")


@app.command()
def figures(
    db: Path = typer.Option(DEFAULT_DB),
    model: str = typer.Option("logistic"),
    out: Path = typer.Option(Path("artifacts/figures")),
) -> None:
    """Render the three result figures for the README."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    out.mkdir(parents=True, exist_ok=True)

    curves: dict[int, dict[str, tuple[float, float, float]]] = {}
    for cut in CUT_POINTS:
        data = load_dataset(str(db), cut)
        results, _ = cross_validate(data)
        curves[cut] = {
            r.name: (r.auroc, r.auroc_ci_lower, r.auroc_ci_upper) for r in results
        }
        typer.echo(f"  cut {cut}: {len(results)} models scored")

    auroc_by_cut_plot(curves, out / "auroc_by_cut.png")
    typer.echo(f"wrote {out / 'auroc_by_cut.png'}")

    data = load_dataset(str(db), GATE_CUT_POINT)
    predictions = out_of_fold_predictions(data, model)

    calibrator = Calibrator().fit(predictions, data.y_fail)
    adjusted = calibrator.apply(predictions)
    calibration_plot(
        raw=calibration(data, predictions),
        calibrated=calibration(data, adjusted),
        path=out / "calibration.png",
        raw_brier=brier_score(data.y_fail, predictions),
        calibrated_brier=brier_score(data.y_fail, adjusted),
    )
    typer.echo(f"wrote {out / 'calibration.png'}")

    estimates = load_token_estimates(str(db), GATE_CUT_POINT)
    points = simulate(data.session_ids, data.y_fail, predictions, estimates)
    savings_tradeoff_plot(points, out / "savings_tradeoff.png")
    typer.echo(f"wrote {out / 'savings_tradeoff.png'}")


@app.command()
def savings(
    db: Path = typer.Option(DEFAULT_DB),
    cut: int = typer.Option(GATE_CUT_POINT),
    model: str = typer.Option("logistic"),
    out: Path = typer.Option(Path("artifacts/phase4")),
) -> None:
    """Simulate token savings against sessions wrongly terminated."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    data = load_dataset(str(db), cut)
    predictions = out_of_fold_predictions(data, model)
    estimates = load_token_estimates(str(db), cut)

    points = simulate(data.session_ids, data.y_fail, predictions, estimates)
    typer.echo(f"cut point {cut}, model {model}, {len(data)} sessions\n")
    typer.echo(render_savings(points))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"savings_cut{cut}.json", "w") as fh:
        json.dump(
            {"cut_point": cut, "model": model, "points": [p.as_dict() for p in points]},
            fh,
            indent=2,
            default=float,
        )
    typer.echo(f"\nwrote {out / f'savings_cut{cut}.json'}")


@app.command()
def site(
    artifacts: Path = typer.Option(Path("artifacts")),
    out: Path = typer.Option(Path("site/index.html"), help="page to write"),
    inline: bool = typer.Option(
        False, "--inline", help="embed figures as data URIs for one portable file"
    ),
) -> None:
    """Render a results page from the committed artifacts."""
    path = build_site(artifacts, out, inline=inline)
    size = path.stat().st_size
    mode = "self-contained" if inline else "figures linked from artifacts/"
    typer.echo(f"wrote {path} ({size / 1024:.0f} KB, {mode}, no JS)")
    typer.echo(f"open it with: open {path}")


@app.command()
def drift(
    db: Path = typer.Option(DEFAULT_DB),
    cut: int = typer.Option(40, help="prefix length at which to compare"),
    out: Path = typer.Option(Path("artifacts/phase6")),
) -> None:
    """Compare feature distributions between the corpus and local sessions."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    paths = discover_transcripts()
    if not paths:
        raise typer.BadParameter("no local transcripts under ~/.claude/projects")

    rows = []
    used = []
    for path in paths:
        transcript = read_transcript(path)
        if len(transcript) < cut:
            continue
        values = extract(transcript.turns[:cut])
        rows.append([values[name] for name in FEATURE_NAMES])
        used.append(transcript.session_id)

    if not rows:
        raise typer.BadParameter(
            f"no local session reached {cut} turns; try a smaller --cut"
        )

    corpus = load_dataset(str(db), cut)
    report = compare(FEATURE_NAMES, corpus.X, np.array(rows, dtype=float))

    typer.echo(report.render())
    typer.echo(f"\nlocal sessions used: {', '.join(s[:8] for s in used)}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"drift_cut{cut}.json", "w") as fh:
        json.dump(
            {
                "cut_point": cut,
                "n_corpus": report.n_corpus,
                "n_local": report.n_local,
                "underpowered": report.underpowered,
                "sessions": used,
                "shifts": [vars(s) for s in report.shifts],
            },
            fh,
            indent=2,
            default=float,
        )
    typer.echo(f"wrote {out / f'drift_cut{cut}.json'}")


@app.command()
def fit(
    db: Path = typer.Option(DEFAULT_DB),
    model: str = typer.Option("logistic"),
    path: Path = typer.Option(DEFAULT_MODEL_PATH, help="where to write the model"),
) -> None:
    """Train on the full corpus at one cut point and persist the model."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")

    data = load_pooled(str(db), CUT_POINTS)

    # Class weighting leaves raw scores on a re-balanced scale, so the
    # calibrator is fitted on out-of-fold predictions before the final fit.
    oof = out_of_fold_predictions(data, model)
    scorer = fit_and_save(
        data.X,
        data.y_fail,
        cut_points=CUT_POINTS,
        model_name=model,
        path=path,
        calibration_scores=oof,
        calibration_labels=data.y_fail,
    )

    raw_brier = brier_score(data.y_fail, oof)
    calibrated_brier = brier_score(data.y_fail, scorer.calibrator.apply(oof))

    size = path.stat().st_size
    typer.echo(f"trained {model} on {len(data)} pooled rows across cuts {list(CUT_POINTS)}")
    typer.echo(
        f"Brier score {raw_brier:.4f} raw, {calibrated_brier:.4f} after isotonic "
        f"calibration ({(1 - calibrated_brier / raw_brier):.1%} better)"
    )
    typer.echo(f"wrote {path} ({size / 1024:.1f} KB, {len(scorer.feature_names)} features)")
    typer.echo(
        "\nNote: cross-validated performance is in artifacts/phase3. This model "
        "is fit on all rows and has no held-out estimate of its own."
    )


@app.command()
def sessions(limit: int = typer.Option(10, help="how many recent transcripts to list")) -> None:
    """List local Claude Code transcripts available for scoring."""
    paths = discover_transcripts()
    if not paths:
        typer.echo("no transcripts found under ~/.claude/projects")
        return

    typer.echo(f"{'session':<40}{'turns':>7}{'tokens':>10}  project")
    for path in paths[:limit]:
        transcript = read_transcript(path)
        typer.echo(
            f"{transcript.session_id:<40}{len(transcript):>7}"
            f"{transcript.total_tokens:>10}  {transcript.repo}"
        )


@app.command()
def score(
    session: str = typer.Argument(None, help="session id; defaults to most recent"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH),
    top: int = typer.Option(5, help="how many contributing features to show"),
) -> None:
    """Score a local Claude Code session for failure risk."""
    paths = discover_transcripts()
    if not paths:
        raise typer.BadParameter("no transcripts found under ~/.claude/projects")

    if session:
        matches = [p for p in paths if p.stem.startswith(session)]
        if not matches:
            raise typer.BadParameter(f"no transcript matching {session!r}")
        chosen = matches[0]
    else:
        chosen = paths[0]

    transcript = read_transcript(chosen)
    scorer = Scorer.load(model_path)
    report = scorer.score(transcript.session_id, transcript.turns, top=top)

    typer.echo(report.render())
    typer.echo(f"\nproject: {transcript.repo}")
    typer.echo(f"tokens so far: {transcript.total_tokens:,}")
    if transcript.malformed_lines:
        typer.echo(f"skipped {transcript.malformed_lines} unparseable lines")


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
