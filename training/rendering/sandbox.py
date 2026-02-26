"""Sandbox environment for safe Manim code execution."""

import os
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxConfig:
    timeout: int = 120
    quality: str = "-ql"
    format: str = "mp4"
    cleanup: bool = True


class RenderSandbox:
    """Creates isolated temp directories for Manim rendering."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._temp_dir: str | None = None

    def __enter__(self) -> "RenderSandbox":
        self._temp_dir = tempfile.mkdtemp(prefix="manim_render_")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.cleanup and self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        return False

    @property
    def temp_dir(self) -> str:
        if self._temp_dir is None:
            raise RuntimeError("Sandbox not initialized. Use as context manager.")
        return self._temp_dir

    def write_scene(self, code: str, filename: str = "scene.py") -> str:
        """Write code to a file in the sandbox and return its path."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(code)
        return filepath

    def get_output_path(self, class_name: str = "GenScene") -> str:
        """Return the expected output video path."""
        return os.path.join(self.temp_dir, f"{class_name}.mp4")

    def find_video(self, class_name: str = "GenScene") -> str | None:
        """Search for rendered video file in sandbox."""
        # Check direct output
        direct = self.get_output_path(class_name)
        if os.path.exists(direct):
            return direct

        # Search recursively for any mp4
        for root, dirs, files in os.walk(self.temp_dir):
            for f in files:
                if f.endswith(".mp4"):
                    return os.path.join(root, f)
        return None
