"""Adapter for SWE-Gym/OpenHands-Sampled-Trajectories.

Records hold OpenAI-style `messages` plus a boolean `resolved` label. The
repository key is encoded in `instance_id`; there is no explicit repo field and
no per-message token accounting.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from averta.normalize import (
    digest,
    error_signature,
    is_valid_json,
    looks_like_error,
    parse_instance_id,
    trajectory_hash,
)
from averta.schema import CONTENT_HEAD_CHARS, Session, Turn

OBSERVATION_ROLES = {"tool", "function"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(part for part in parts if part)
    return ""


def _agent_from_run_id(run_id: str | None) -> str | None:
    if not run_id:
        return None
    return run_id.split("_maxiter")[0]


class SweGymAdapter:
    source = "swegym"
    repo_id = "SWE-Gym/OpenHands-Sampled-Trajectories"
    split = "train.raw"

    def load(self, streaming: bool = True) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        return iter(load_dataset(self.repo_id, split=self.split, streaming=streaming))

    def convert(self, record: dict[str, Any]) -> Session | None:
        instance_id = record.get("instance_id")
        resolved = record.get("resolved")
        if not instance_id or resolved is None:
            return None

        repo, _ = parse_instance_id(instance_id)
        if repo is None:
            return None

        run_id = record.get("run_id")
        messages = record.get("messages") or []

        # The dataset contains records sharing an instance and run id that are
        # nevertheless different trajectories. Keying on content keeps those
        # distinct while still collapsing genuine duplicates.
        fingerprint = trajectory_hash(
            f"{message.get('role')}|{_text(message.get('content'))[:200]}"
            for message in messages
        )
        session_id = f"{self.source}:{instance_id}:{run_id}:{fingerprint[:8]}"

        turns = list(self._turns(session_id, messages))
        if not turns:
            return None

        report = (record.get("test_result") or {}).get("report") or {}

        return Session(
            session_id=session_id,
            source=self.source,
            repo=repo,
            instance_id=instance_id,
            run_id=run_id,
            agent=_agent_from_run_id(run_id),
            outcome=bool(resolved),
            n_turns=len(turns),
            n_steps=sum(1 for turn in turns if turn.step_index is not None),
            trajectory_hash=fingerprint,
            turns=turns,
            metadata={
                "empty_generation": report.get("empty_generation"),
                "error_eval": report.get("error_eval"),
                "failed_apply_patch": report.get("failed_apply_patch"),
                "has_patch": bool((record.get("test_result") or {}).get("git_patch")),
            },
        )

    def _turns(self, session_id: str, messages: list[dict[str, Any]]) -> Iterator[Turn]:
        step = 0
        for index, message in enumerate(messages):
            role = message.get("role") or "unknown"
            content = _text(message.get("content"))
            tool_calls = message.get("tool_calls") or []

            tool_name = None
            tool_input = None
            tool_input_hash = None
            tool_input_bad = False
            if tool_calls:
                function = tool_calls[0].get("function") or {}
                tool_name = function.get("name")
                arguments = function.get("arguments")
                tool_input = arguments if isinstance(arguments, str) else json.dumps(arguments)
                tool_input_hash = digest(tool_input)
                tool_input_bad = not is_valid_json(tool_input)

            step_index = None
            if role == "assistant":
                step_index = step
                step += 1

            is_error = role in OBSERVATION_ROLES and looks_like_error(content)

            yield Turn(
                session_id=session_id,
                turn_index=index,
                step_index=step_index,
                role=role,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_input_hash=tool_input_hash,
                tool_input_bad=tool_input_bad,
                n_tool_calls=len(tool_calls),
                content_chars=len(content),
                content_head=content[:CONTENT_HEAD_CHARS] or None,
                is_error=is_error,
                error_signature=error_signature(content) if is_error else None,
            )
