"""Core Manim verifier: runs Manim subprocess and returns structured results."""

import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum

from .sandbox import RenderSandbox, SandboxConfig


class ErrorType(str, Enum):
    NONE = "none"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    CLASS_NOT_FOUND = "class_not_found"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class VerifyResult:
    success: bool
    error_type: ErrorType = ErrorType.NONE
    error_message: str = ""
    animation_count: int = 0
    render_time: float = 0.0
    code: str = ""


def extract_class_name(code: str) -> str | None:
    """Extract the Scene subclass name from Manim code."""
    match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
    return match.group(1) if match else None


def classify_error(stderr: str) -> ErrorType:
    """Classify Manim error from stderr output."""
    if "SyntaxError" in stderr:
        return ErrorType.SYNTAX_ERROR
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return ErrorType.IMPORT_ERROR
    if "is not in the script" in stderr:
        return ErrorType.CLASS_NOT_FOUND
    if "Traceback" in stderr:
        return ErrorType.RUNTIME_ERROR
    return ErrorType.UNKNOWN


def count_animations(stderr: str) -> int:
    """Count animations from Manim progress output."""
    matches = re.findall(r"Animation\s+(\d+):", stderr)
    if not matches:
        return 0
    return max(int(m) for m in matches) + 1


def ensure_imports(code: str) -> str:
    """Ensure code has required imports."""
    lines = code.strip().split("\n")
    has_manim_import = any("from manim import" in line for line in lines)
    if not has_manim_import:
        code = "from manim import *\n" + code
    return code


def verify_code(
    code: str,
    timeout: int = 120,
    quality: str = "-ql",
) -> VerifyResult:
    """
    Render Manim code in a sandbox and return structured result.

    Replicates the rendering pattern from /api/routes/video_rendering.py:
    - Creates temp directory
    - Writes code to scene.py with imports
    - Extracts class name via regex
    - Runs: manim scene.py ClassName --format=mp4 -ql --media_dir . --custom_folders
    - Parses stderr for animation progress
    - Classifies errors
    - Cleans up
    """
    code = ensure_imports(code)

    # Extract class name
    class_name = extract_class_name(code)
    if not class_name:
        return VerifyResult(
            success=False,
            error_type=ErrorType.CLASS_NOT_FOUND,
            error_message="No Scene subclass found in code",
            code=code,
        )

    config = SandboxConfig(timeout=timeout, quality=quality)

    with RenderSandbox(config) as sandbox:
        scene_path = sandbox.write_scene(code)
        start_time = time.time()

        command = [
            "manim",
            scene_path,
            class_name,
            f"--format=mp4",
            quality,
            "--media_dir",
            sandbox.temp_dir,
            "--custom_folders",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox.temp_dir,
            )
            render_time = time.time() - start_time
            stderr = result.stderr or ""

            if result.returncode == 0:
                return VerifyResult(
                    success=True,
                    animation_count=count_animations(stderr),
                    render_time=render_time,
                    code=code,
                )
            else:
                return VerifyResult(
                    success=False,
                    error_type=classify_error(stderr),
                    error_message=stderr[-500:] if len(stderr) > 500 else stderr,
                    animation_count=count_animations(stderr),
                    render_time=render_time,
                    code=code,
                )

        except subprocess.TimeoutExpired:
            render_time = time.time() - start_time
            return VerifyResult(
                success=False,
                error_type=ErrorType.TIMEOUT,
                error_message=f"Rendering timed out after {timeout}s",
                render_time=render_time,
                code=code,
            )
        except Exception as e:
            render_time = time.time() - start_time
            return VerifyResult(
                success=False,
                error_type=ErrorType.UNKNOWN,
                error_message=str(e),
                render_time=render_time,
                code=code,
            )


def batch_verify(
    codes: list[str],
    max_workers: int = 4,
    timeout: int = 120,
    quality: str = "-ql",
) -> list[VerifyResult]:
    """Verify multiple Manim codes in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = [None] * len(codes)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(verify_code, code, timeout, quality): i
            for i, code in enumerate(codes)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = VerifyResult(
                    success=False,
                    error_type=ErrorType.UNKNOWN,
                    error_message=str(e),
                    code=codes[idx],
                )

    return results
