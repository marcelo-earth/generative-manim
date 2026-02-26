"""Evaluate model outputs: render success rate, error breakdown, animation stats."""

import argparse
import json
from collections import Counter
from pathlib import Path

import wandb
from rich.console import Console
from rich.table import Table

from ..rendering.manim_verifier import verify_code, ErrorType
from ..utils.code_extraction import clean_code
from ..utils.logging_utils import setup_wandb, log_metrics, log_table, finish_wandb

console = Console()


def evaluate_responses(
    responses_path: str | Path,
    model_name: str = "unknown",
    stage: str = "unknown",
    max_workers: int = 4,
    timeout: int = 120,
    log_wandb: bool = True,
) -> dict:
    """
    Evaluate a set of model responses by rendering each with Manim.

    Returns metrics dict with:
    - render_success_rate
    - error_type_breakdown
    - avg_animation_count
    - avg_render_time
    """
    responses_path = Path(responses_path)

    # Load responses
    responses = []
    with open(responses_path) as f:
        for line in f:
            responses.append(json.loads(line))

    print(f"Evaluating {len(responses)} responses ({model_name}/{stage})...")

    # Verify each response
    results = []
    error_counts = Counter()
    total_animations = 0
    total_render_time = 0.0
    successes = 0

    from ..rendering.manim_verifier import batch_verify

    codes = [clean_code(r["response"]) for r in responses]
    verify_results = batch_verify(codes, max_workers=max_workers, timeout=timeout)

    for response, result in zip(responses, verify_results):
        results.append({
            "prompt": response["prompt"],
            "success": result.success,
            "error_type": result.error_type.value,
            "animation_count": result.animation_count,
            "render_time": result.render_time,
        })

        if result.success:
            successes += 1
            total_animations += result.animation_count
        else:
            error_counts[result.error_type.value] += 1
        total_render_time += result.render_time

    # Compute metrics
    n = len(responses)
    metrics = {
        "render_success_rate": successes / n if n else 0,
        "total_evaluated": n,
        "successes": successes,
        "failures": n - successes,
        "avg_animation_count": total_animations / successes if successes else 0,
        "avg_render_time": total_render_time / n if n else 0,
        "error_breakdown": dict(error_counts),
    }

    # Display results
    table = Table(title=f"Evaluation: {model_name} / {stage}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Render Success Rate", f"{metrics['render_success_rate']:.1%}")
    table.add_row("Successes / Total", f"{successes} / {n}")
    table.add_row("Avg Animations", f"{metrics['avg_animation_count']:.1f}")
    table.add_row("Avg Render Time", f"{metrics['avg_render_time']:.1f}s")

    for error_type, count in sorted(error_counts.items()):
        table.add_row(f"  {error_type}", str(count))

    console.print(table)

    # Log to W&B
    if log_wandb and wandb.run is not None:
        log_metrics({
            f"eval/{model_name}/{stage}/render_success_rate": metrics["render_success_rate"],
            f"eval/{model_name}/{stage}/avg_animations": metrics["avg_animation_count"],
        })

        # Log detailed results as table
        columns = ["prompt", "success", "error_type", "animations", "render_time"]
        data = [
            [r["prompt"][:80], r["success"], r["error_type"], r["animation_count"], round(r["render_time"], 1)]
            for r in results
        ]
        log_table(f"eval_details/{model_name}/{stage}", columns, data)

    # Save detailed results
    output_path = responses_path.parent / f"eval_results_{model_name}_{stage}.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses")
    parser.add_argument("--responses", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="unknown")
    parser.add_argument("--stage", type=str, default="unknown")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    evaluate_responses(
        responses_path=args.responses,
        model_name=args.model_name,
        stage=args.stage,
        max_workers=args.workers,
        timeout=args.timeout,
        log_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
