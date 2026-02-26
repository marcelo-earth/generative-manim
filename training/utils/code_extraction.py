"""Extract Python code from LLM responses (markdown fences, etc.)."""

import re


def extract_python_code(text: str) -> str:
    """
    Extract Python code from a response that may contain markdown fences.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Raw code without fences
    """
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to find ``` ... ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no fences, return the text as-is (likely raw code)
    return text.strip()


def normalize_class_name(code: str, target: str = "GenScene") -> str:
    """Replace any Scene subclass name with the target name."""
    pattern = r"class\s+(\w+)\s*\((.*?Scene.*?)\)"
    match = re.search(pattern, code)
    if match and match.group(1) != target:
        code = code.replace(match.group(1), target)
    return code


def ensure_manim_import(code: str) -> str:
    """Ensure code starts with manim import."""
    if "from manim import" not in code:
        code = "from manim import *\n" + code
    return code


def clean_code(code: str) -> str:
    """Full cleaning pipeline: extract, normalize, ensure imports."""
    code = extract_python_code(code)
    code = normalize_class_name(code)
    code = ensure_manim_import(code)
    return code
