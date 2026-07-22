#!/usr/bin/env python3
"""Audit a LeRobot/LIBERO dataset and build non-destructive clean manifests.

The script never rewrites source parquet files. Episodes that satisfy the
configured schema and numeric checks are listed in ``accepted_episodes.jsonl``;
flagged episodes are retained in ``rejected_episodes.jsonl`` with reasons.
This makes the filtering decision explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FIELDS = (
    "image",
    "wrist_image",
    "state",
    "actions",
    "frame_index",
    "episode_index",
    "task_index",
)


@dataclass(frozen=True)
class AuditConfig:
    expected_state_dim: int = 8
    expected_action_dim: int = 7
    min_episode_length: int = 50
    action_abs_limit: float = 1.05
    state_abs_limit: float = 10.0
    all_zero_action_ratio_limit: float = 0.95
    zero_action_epsilon: float = 1e-6


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_tasks(meta_dir: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row in read_jsonl(meta_dir / "tasks.jsonl"):
        index = row.get("task_index", row.get("index"))
        text = row.get("task", row.get("task_text", row.get("text")))
        if index is not None and text:
            tasks[int(index)] = str(text)
    return tasks


def flatten_numeric(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"Expected numeric value, got {type(item).__name__}")
        flattened.append(float(item))

    visit(value)
    return flattened


def image_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return value.get("bytes") is None and value.get("path") is None
    return False


def _add_once(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def validate_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_file: str,
    tasks: Mapping[int, str],
    config: AuditConfig,
) -> dict[str, Any]:
    """Validate one episode. Pure-Python so core behavior is unit-testable."""

    flags: list[str] = []
    if not rows:
        return {
            "episode_file": episode_file,
            "episode_index": None,
            "task_index": None,
            "task_text": None,
            "num_frames": 0,
            "accepted": False,
            "flags": ["empty_episode"],
        }

    if len(rows) < config.min_episode_length:
        _add_once(flags, "short_episode")

    first = rows[0]
    missing_fields = [field for field in REQUIRED_FIELDS if field not in first]
    for field in missing_fields:
        _add_once(flags, f"missing_field:{field}")

    episode_index = first.get("episode_index")
    task_index = first.get("task_index")
    task_text = tasks.get(int(task_index)) if task_index is not None else None
    if task_text is None:
        _add_once(flags, "unmapped_task")

    zero_actions = 0
    previous_timestamp: float | None = None
    state_min = math.inf
    state_max = -math.inf
    action_min = math.inf
    action_max = -math.inf

    for offset, row in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if field not in row:
                _add_once(flags, f"missing_field:{field}")

        if row.get("episode_index") != episode_index:
            _add_once(flags, "mixed_episode_index")
        if row.get("task_index") != task_index:
            _add_once(flags, "mixed_task_index")
        if row.get("frame_index") != offset:
            _add_once(flags, "non_contiguous_frame_index")
        if image_is_missing(row.get("image")):
            _add_once(flags, "missing_main_image")
        if image_is_missing(row.get("wrist_image")):
            _add_once(flags, "missing_wrist_image")

        timestamp = row.get("timestamp")
        if timestamp is not None:
            try:
                timestamp_value = float(timestamp)
                if not math.isfinite(timestamp_value):
                    _add_once(flags, "non_finite_timestamp")
                if previous_timestamp is not None and timestamp_value < previous_timestamp:
                    _add_once(flags, "non_monotonic_timestamp")
                previous_timestamp = timestamp_value
            except (TypeError, ValueError):
                _add_once(flags, "invalid_timestamp")

        try:
            state = flatten_numeric(row.get("state"))
            if len(state) != config.expected_state_dim:
                _add_once(flags, "invalid_state_dim")
            if not all(math.isfinite(value) for value in state):
                _add_once(flags, "non_finite_state")
            if state:
                state_min = min(state_min, min(state))
                state_max = max(state_max, max(state))
                if max(abs(value) for value in state) > config.state_abs_limit:
                    _add_once(flags, "state_out_of_range")
        except (TypeError, ValueError):
            _add_once(flags, "invalid_state")

        try:
            action = flatten_numeric(row.get("actions"))
            if len(action) != config.expected_action_dim:
                _add_once(flags, "invalid_action_dim")
            if not all(math.isfinite(value) for value in action):
                _add_once(flags, "non_finite_action")
            if action:
                action_min = min(action_min, min(action))
                action_max = max(action_max, max(action))
                if max(abs(value) for value in action) > config.action_abs_limit:
                    _add_once(flags, "action_out_of_range")
                if math.sqrt(sum(value * value for value in action)) < config.zero_action_epsilon:
                    zero_actions += 1
        except (TypeError, ValueError):
            _add_once(flags, "invalid_action")

    zero_action_ratio = zero_actions / len(rows)
    if zero_action_ratio > config.all_zero_action_ratio_limit:
        _add_once(flags, "mostly_zero_actions")

    return {
        "episode_file": episode_file,
        "episode_index": int(episode_index) if episode_index is not None else None,
        "task_index": int(task_index) if task_index is not None else None,
        "task_text": task_text,
        "num_frames": len(rows),
        "state_dim": config.expected_state_dim,
        "action_dim": config.expected_action_dim,
        "state_range": None if state_min == math.inf else [state_min, state_max],
        "action_range": None if action_min == math.inf else [action_min, action_max],
        "zero_action_ratio": zero_action_ratio,
        "accepted": not flags,
        "flags": sorted(flags),
    }


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the Hugging Face 'datasets' package to read parquet episodes") from exc
    dataset = load_dataset("parquet", data_files=[str(path)], split="train")
    return [dataset[index] for index in range(len(dataset))]


def parse_episode_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/data_quality/latest"))
    parser.add_argument("--episode-ids", help="Comma-separated episode indices; omit to scan all")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means no limit")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any episode is rejected")
    parser.add_argument("--min-episode-length", type=int, default=50)
    parser.add_argument("--action-abs-limit", type=float, default=1.05)
    parser.add_argument("--state-abs-limit", type=float, default=10.0)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    meta_dir = root / "meta"
    if not (meta_dir / "tasks.jsonl").exists():
        raise FileNotFoundError(f"Missing {meta_dir / 'tasks.jsonl'}")

    selected_ids = parse_episode_ids(args.episode_ids)
    episode_files = sorted(root.rglob("episode_*.parquet"))
    if selected_ids is not None:
        episode_files = [
            path for path in episode_files if int(path.stem.removeprefix("episode_")) in selected_ids
        ]
    if args.max_episodes > 0:
        episode_files = episode_files[: args.max_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.parquet files selected under {root}")

    config = AuditConfig(
        min_episode_length=args.min_episode_length,
        action_abs_limit=args.action_abs_limit,
        state_abs_limit=args.state_abs_limit,
    )
    tasks = load_tasks(meta_dir)
    reports = [
        validate_episode_rows(
            load_parquet_rows(path), episode_file=path.name, tasks=tasks, config=config
        )
        for path in episode_files
    ]
    accepted = [report for report in reports if report["accepted"]]
    rejected = [report for report in reports if not report["accepted"]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "accepted_episodes.jsonl", accepted)
    write_jsonl(output_dir / "rejected_episodes.jsonl", rejected)

    summary = {
        "schema_version": "libero_data_audit_v1",
        "source_kind": "LeRobot parquet episodes",
        "source_mutated": False,
        "config": asdict(config),
        "num_tasks": len(tasks),
        "num_selected_episodes": len(reports),
        "num_accepted_episodes": len(accepted),
        "num_rejected_episodes": len(rejected),
        "num_checked_frames": sum(report["num_frames"] for report in reports),
        "rejection_counts": {
            flag: sum(flag in report["flags"] for report in rejected)
            for flag in sorted({flag for report in rejected for flag in report["flags"]})
        },
        "manifests": {
            "accepted": "accepted_episodes.jsonl",
            "rejected": "rejected_episodes.jsonl",
        },
    }
    with (output_dir / "data_quality_report.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and rejected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
