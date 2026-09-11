"""Inspect the structure of candidate trajectory datasets without downloading them."""

from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import DatasetCard, HfApi

SOURCES = {
    "swegym": "SWE-Gym/OpenHands-Sampled-Trajectories",
    "nebius": "nebius/SWE-rebench-openhands-trajectories",
}

MAX_PREVIEW = 160


def describe(value: Any, depth: int = 0) -> str:
    indent = "  " * depth
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = [f"{indent}  {k}: {describe(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + "\n".join(lines) + f"\n{indent}}}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} items] first -> {describe(value[0], depth + 1)}"
    if isinstance(value, str):
        preview = value.replace("\n", "\\n")
        if len(preview) > MAX_PREVIEW:
            preview = preview[:MAX_PREVIEW] + "..."
        return f"str({len(value)}) {preview!r}"
    return f"{type(value).__name__}({value!r})"


def show_card(repo_id: str) -> None:
    api = HfApi()
    info = api.dataset_info(repo_id)
    print(f"  license tag  : {info.card_data.get('license') if info.card_data else 'unknown'}")
    print(f"  downloads    : {info.downloads}")
    print(f"  last modified: {info.last_modified}")
    configs = {s.config_name for s in info.siblings or [] if hasattr(s, "config_name")}
    if configs:
        print(f"  configs      : {sorted(configs)}")

    try:
        card = DatasetCard.load(repo_id)
        text = card.text.strip()
        if text:
            print("  card excerpt :")
            for line in text.splitlines()[:12]:
                print(f"    {line}")
    except Exception as exc:  # card is optional metadata
        print(f"  card         : unavailable ({exc})")


def resolve_target(repo_id: str) -> tuple[str | None, str]:
    configs = get_dataset_config_names(repo_id)
    config = configs[0] if configs else None
    splits = get_dataset_split_names(repo_id, config_name=config)
    print(f"  configs      : {configs}")
    print(f"  splits       : {splits}")
    return config, splits[0]


def probe(name: str, repo_id: str, n: int) -> None:
    print(f"\n{'=' * 78}\n{name}: {repo_id}\n{'=' * 78}")

    try:
        show_card(repo_id)
    except Exception as exc:
        print(f"  metadata unavailable: {exc}")

    try:
        config, split = resolve_target(repo_id)
        stream = load_dataset(repo_id, name=config, split=split, streaming=True)
    except Exception as exc:
        print(f"\n  FAILED to open stream: {exc}")
        return

    for i, record in enumerate(stream):
        if i >= n:
            break
        print(f"\n  --- record {i} ---")
        for key, value in record.items():
            print(f"  {key}: {describe(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--dump", metavar="PATH", help="write the first record as JSON")
    args = parser.parse_args()

    targets = SOURCES if args.source == "all" else {args.source: SOURCES[args.source]}

    for name, repo_id in targets.items():
        probe(name, repo_id, args.records)

    if args.dump:
        name, repo_id = next(iter(targets.items()))
        stream = load_dataset(repo_id, split="train", streaming=True)
        record = next(iter(stream))
        with open(args.dump, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
        print(f"\nwrote first {name} record to {args.dump}")


if __name__ == "__main__":
    main()
