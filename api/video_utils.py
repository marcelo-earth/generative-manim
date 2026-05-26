"""Shared utilities for video generation and rendering routes."""

from typing import Tuple


def get_frame_config(aspect_ratio: str) -> Tuple[Tuple[int, int], float]:
    """Return (frame_size, frame_width) for a given aspect ratio string."""
    if aspect_ratio == "9:16":
        return (1080, 1920), 8.0
    if aspect_ratio == "1:1":
        return (1080, 1080), 8.0
    return (3840, 2160), 14.22
