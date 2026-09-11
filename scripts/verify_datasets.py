"""Measure label coverage, turn distributions and grouping keys for candidate datasets.

Phase 0 gate: establishes whether the public trajectory datasets carry usable
outcome labels, repository identifiers for grouped splits, and per-step token
counts. Writes a JSON report alongside a readable summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi

SOURCES = {
    "swegym": "SWE-Gym/OpenHands-Sampled-Trajectories",
    "nebius": "nebius/SWE-rebench-openhands-trajectories",
}

INSTANCE_PATTERN = re.compile(r"^(?P<owner>[^_]+(?:_[^_]+)*)__(?P<repo>.+)-(?P<number>\d+)$")

TOKEN_FIELD_HINTS = ("token", "usage", "n_tokens", "prompt_tokens", "completion_tokens")


def parse_instance_id(instance_id: str) -> tuple[str | None, str | None]:
    match = INSTANCE_PATTERN.match(instance_id)
    if not match:
        return None, None
    return f"{match['owner']}/{match['repo']}", match["number"]


def find_token_fields(record: dict[str, Any]) -> list[str]:
    found = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}" if path else key
                if any(hint in key.lower() for hint in TOKEN_FIELD_HINTS):
                    found.append(child)
                walk(value, child)
        elif isinstance(node, list) and node:
            walk(node[0], f"{path}[]")

    walk(record, "")
    return found


def message_stats(messages: list[dict[str, Any]]) -> tuple[int, Counter, int]:
    tools = Counter()
    errors = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if name := function.get("name"):
                tools[name] += 1
        content = message.get("content") or ""
        if isinstance(content, str) and "error" in content[:400].lower():
            errors += 1
    return len(messages), tools, errors


def collect(repo_id: str, limit: int | None) -> dict[str, Any]:
    configs = get_dataset_config_names(repo_id)
    config = configs[0] if configs else None
    split = get_dataset_split_names(repo_id, config_name=config)[0]

    info = HfApi().dataset_info(repo_id)
    declared_rows = None
    if info.card_data and (infos := info.card_data.get("dataset_info")):
        entries = infos if isinstance(infos, list) else [infos]
        for entry in entries:
            for split_info in entry.get("splits", []):
                if split_info.get("name") == split:
                    declared_rows = split_info.get("num_examples")

    stream = load_dataset(repo_id, name=config, split=split, streaming=True)

    rows: list[dict[str, Any]] = []
    tool_vocab: Counter = Counter()
    token_fields: set[str] = set()
    seen_keys: set[str] = set()

    for i, record in enumerate(stream):
        if limit is not None and i >= limit:
            break
        seen_keys.update(record.keys())
        if i == 0:
            token_fields.update(find_token_fields(record))

        instance_id = record.get("instance_id")
        repo, number = parse_instance_id(instance_id) if instance_id else (None, None)
        messages = record.get("messages") or []
        n_turns, tools, errors = message_stats(messages)
        tool_vocab.update(tools)

        rows.append(
            {
                "instance_id": instance_id,
                "run_id": record.get("run_id"),
                "repo": repo,
                "issue_number": number,
                "resolved": record.get("resolved"),
                "n_turns": n_turns,
                "n_tool_calls": sum(tools.values()),
                "n_error_messages": errors,
            }
        )

    frame = pl.DataFrame(rows)

    return {
        "repo_id": repo_id,
        "config": config,
        "split": split,
        "license": (info.card_data.get("license") if info.card_data else None),
        "last_modified": str(info.last_modified),
        "declared_rows": declared_rows,
        "scanned_rows": frame.height,
        "truncated": limit is not None and frame.height >= limit,
        "record_keys": sorted(seen_keys),
        "token_fields": sorted(token_fields),
        "frame": frame,
        "tool_vocab": tool_vocab,
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    frame: pl.DataFrame = result["frame"]
    tool_vocab: Counter = result["tool_vocab"]

    resolved = frame["resolved"]
    turns = frame["n_turns"]

    report = {
        "repo_id": result["repo_id"],
        "config": result["config"],
        "split": result["split"],
        "license": result["license"],
        "last_modified": result["last_modified"],
        "declared_rows": result["declared_rows"],
        "scanned_rows": result["scanned_rows"],
        "truncated": result["truncated"],
        "record_keys": result["record_keys"],
        "token_fields": result["token_fields"],
        "label": {
            "field": "resolved",
            "non_null": int(resolved.is_not_null().sum()),
            "coverage": round(resolved.is_not_null().mean() or 0.0, 4),
            "resolve_rate": round(resolved.mean() or 0.0, 4),
        },
        "grouping": {
            "distinct_repos": frame["repo"].n_unique(),
            "unparsed_instance_ids": int(frame["repo"].is_null().sum()),
            "distinct_instances": frame["instance_id"].n_unique(),
            "distinct_runs": frame["run_id"].n_unique(),
            "duplicate_instance_run_pairs": frame.height
            - frame.select(["instance_id", "run_id"]).unique().height,
        },
        "turns": {
            "min": int(turns.min()),
            "p25": int(turns.quantile(0.25)),
            "median": int(turns.median()),
            "p75": int(turns.quantile(0.75)),
            "max": int(turns.max()),
            "mean": round(float(turns.mean()), 2),
        },
        "tool_vocab_size": len(tool_vocab),
        "tool_vocab": dict(tool_vocab.most_common()),
    }

    by_outcome = (
        frame.group_by("resolved")
        .agg(pl.len().alias("n"), pl.col("n_turns").mean().round(2).alias("mean_turns"))
        .sort("resolved")
    )
    report["turns_by_outcome"] = by_outcome.to_dicts()

    return report


def render(report: dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\n{report['repo_id']}\n{'=' * 78}")
    print(f"  split            : {report['config']}/{report['split']}")
    print(f"  license          : {report['license'] or 'NOT DECLARED'}")
    print(f"  declared rows    : {report['declared_rows']}")
    suffix = " (truncated)" if report["truncated"] else ""
    print(f"  scanned rows     : {report['scanned_rows']}{suffix}")
    print(f"  fields           : {', '.join(report['record_keys'])}")

    label = report["label"]
    print(f"\n  label '{label['field']}' coverage : {label['coverage']:.1%}")
    print(f"  resolve rate     : {label['resolve_rate']:.1%}")

    group = report["grouping"]
    print(f"\n  distinct repos   : {group['distinct_repos']}")
    print(f"  distinct instances: {group['distinct_instances']}")
    print(f"  distinct runs    : {group['distinct_runs']}")
    print(f"  unparsed ids     : {group['unparsed_instance_ids']}")
    print(f"  duplicate pairs  : {group['duplicate_instance_run_pairs']}")

    turns = report["turns"]
    print(
        f"\n  turns            : min {turns['min']} / p25 {turns['p25']} / "
        f"median {turns['median']} / p75 {turns['p75']} / max {turns['max']}"
    )
    for row in report["turns_by_outcome"]:
        print(f"    resolved={row['resolved']}: n={row['n']}, mean turns={row['mean_turns']}")

    print(f"\n  tool vocabulary  : {report['tool_vocab_size']} distinct")
    for name, count in list(report["tool_vocab"].items())[:15]:
        print(f"    {name}: {count}")

    if report["token_fields"]:
        print(f"\n  token fields     : {report['token_fields']}")
    else:
        print("\n  token fields     : NONE — savings simulation must estimate from text length")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--limit", type=int, help="stop after N records per source")
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase0"))
    args = parser.parse_args()

    targets = SOURCES if args.source == "all" else {args.source: SOURCES[args.source]}
    args.out.mkdir(parents=True, exist_ok=True)

    for name, repo_id in targets.items():
        result = collect(repo_id, args.limit)
        report = summarize(result)
        render(report)

        result["frame"].write_parquet(args.out / f"{name}_index.parquet")
        with open(args.out / f"{name}_report.json", "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n  wrote {args.out / f'{name}_report.json'}")


if __name__ == "__main__":
    main()
