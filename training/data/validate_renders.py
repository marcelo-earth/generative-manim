"""Validate completions by rendering each with Manim. Keep only successes."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from tqdm import tqdm

from ..rendering.manim_verifier import batch_verify, verify_code
from ..utils.code_extraction import clean_code

console = Console()

DEFAULT_INPUT = Path(__file__).parent / "outputs" / "raw_completions.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "validated_completions.jsonl"


def validate_all(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    max_workers: int = 4,
    timeout: int = 120,
    resume: bool = True,
):
    """Validate all completions by rendering."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load completions
    completions = []
    with open(input_path) as f:
        for line in f:
            completions.append(json.loads(line))

    # Resume support
    validated_prompts = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                validated_prompts.add(data["prompt"])
        console.print(f"[green]Resuming: {len(validated_prompts)} already validated[/]")

    remaining = [c for c in completions if c["prompt"] not in validated_prompts]
    console.print(f"Validating {len(remaining)} completions ({max_workers} workers)...")

    # Process in batches
    batch_size = max_workers * 2
    stats = {"success": 0, "fail": 0, "total": 0}

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        codes = [clean_code(c["completion"]) for c in batch]

        results = batch_verify(codes, max_workers=max_workers, timeout=timeout)

        with open(output_path, "a") as f:
            for completion, result in zip(batch, results):
                stats["total"] += 1
                if result.success:
                    stats["success"] += 1
                    validated = {
                        "category": completion["category"],
                        "prompt": completion["prompt"],
                        "code": result.code,
                        "animation_count": result.animation_count,
                        "render_time": result.render_time,
                    }
                    f.write(json.dumps(validated) + "\n")
                else:
                    stats["fail"] += 1

        rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        console.print(
            f"  [{stats['total']}/{len(remaining)}] "
            f"Success: {stats['success']} | Fail: {stats['fail']} | Rate: {rate:.1f}%"
        )

    final_rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
    console.print(f"\n[bold green]Done![/] Success rate: {final_rate:.1f}%")
    console.print(f"  Validated: {stats['success'] + len(validated_prompts)} total")


def main():
    parser = argparse.ArgumentParser(description="Validate Manim completions")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    validate_all(
        input_path=args.input,
        output_path=args.output,
        max_workers=args.workers,
        timeout=args.timeout,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
