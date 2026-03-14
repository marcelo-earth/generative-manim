"""Utilities for pass@k estimation."""

from __future__ import annotations

from math import comb


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float | None:
    """
    Unbiased pass@k estimator used in code-generation benchmarks.

    Returns None when there are fewer than k samples for the task.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if num_samples < k:
        return None
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - (comb(num_samples - num_correct, k) / comb(num_samples, k))

