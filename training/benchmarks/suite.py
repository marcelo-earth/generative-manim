"""Benchmark task loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import DEFAULT_SUITE


@dataclass
class BenchmarkTask:
    """Frozen benchmark task definition."""

    task_id: str
    category: str
    difficulty: str
    prompt: str
    required_patterns: list[str] = field(default_factory=list)
    disallowed_patterns: list[str] = field(default_factory=list)
    min_animation_count: int = 1
    notes: str = ""


def _validate_task(data: dict, source: Path, line_number: int) -> BenchmarkTask:
    required_keys = {"task_id", "category", "difficulty", "prompt"}
    missing = required_keys - set(data)
    if missing:
        raise ValueError(
            f"{source}:{line_number} is missing required keys: {sorted(missing)}"
        )

    return BenchmarkTask(
        task_id=str(data["task_id"]),
        category=str(data["category"]),
        difficulty=str(data["difficulty"]),
        prompt=str(data["prompt"]),
        required_patterns=list(data.get("required_patterns", [])),
        disallowed_patterns=list(data.get("disallowed_patterns", [])),
        min_animation_count=int(data.get("min_animation_count", 1)),
        notes=str(data.get("notes", "")),
    )


def load_suite(path: str | Path = DEFAULT_SUITE) -> list[BenchmarkTask]:
    """Load a benchmark suite from JSONL."""
    suite_path = Path(path)
    tasks: list[BenchmarkTask] = []
    seen_ids: set[str] = set()

    with open(suite_path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            task = _validate_task(json.loads(line), suite_path, line_number)
            if task.task_id in seen_ids:
                raise ValueError(f"Duplicate task_id in {suite_path}: {task.task_id}")
            tasks.append(task)
            seen_ids.add(task.task_id)

    if not tasks:
        raise ValueError(f"No tasks found in suite: {suite_path}")

    return tasks

