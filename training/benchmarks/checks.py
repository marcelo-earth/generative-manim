"""Static benchmark checks for Manim code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .suite import BenchmarkTask


@dataclass
class StaticCheckResult:
    """Regex-based compliance checks for a benchmark task."""

    required_patterns_total: int
    required_patterns_passed: int
    matched_patterns: list[str]
    missing_patterns: list[str]
    disallowed_matches: list[str]

    @property
    def required_pattern_rate(self) -> float:
        if self.required_patterns_total == 0:
            return 1.0
        return self.required_patterns_passed / self.required_patterns_total


def run_static_checks(task: BenchmarkTask, code: str) -> StaticCheckResult:
    """Evaluate task-specific regex checks against cleaned code."""
    matched_patterns: list[str] = []
    missing_patterns: list[str] = []
    disallowed_matches: list[str] = []

    for pattern in task.required_patterns:
        if re.search(pattern, code, re.MULTILINE):
            matched_patterns.append(pattern)
        else:
            missing_patterns.append(pattern)

    for pattern in task.disallowed_patterns:
        if re.search(pattern, code, re.MULTILINE):
            disallowed_matches.append(pattern)

    return StaticCheckResult(
        required_patterns_total=len(task.required_patterns),
        required_patterns_passed=len(matched_patterns),
        matched_patterns=matched_patterns,
        missing_patterns=missing_patterns,
        disallowed_matches=disallowed_matches,
    )

