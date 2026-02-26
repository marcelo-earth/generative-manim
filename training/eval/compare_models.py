"""Cross-model x cross-stage comparison table."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

MODELS = ["qwen2.5-coder-7b", "deepseek-coder-v2-lite", "codellama-7b"]
STAGES = ["sft", "dpo", "grpo"]


def load_eval_results(results_dir: str | Path) -> dict:
    """Load all evaluation result files from a directory."""
    results_dir = Path(results_dir)
    all_results = {}

    for model in MODELS:
        for stage in STAGES:
            path = results_dir / f"eval_results_{model}_{stage}.jsonl"
            if path.exists():
                entries = []
                with open(path) as f:
                    for line in f:
                        entries.append(json.loads(line))

                successes = sum(1 for e in entries if e["success"])
                total = len(entries)
                rate = successes / total if total else 0

                all_results[(model, stage)] = {
                    "rate": rate,
                    "successes": successes,
                    "total": total,
                }

    return all_results


def compare(results_dir: str | Path = "./outputs"):
    """Generate comparison table."""
    results = load_eval_results(results_dir)

    if not results:
        console.print("[red]No evaluation results found.[/]")
        console.print(f"Looking in: {Path(results_dir).resolve()}")
        return

    # Build comparison table
    table = Table(title="Model Comparison: Render Success Rate")
    table.add_column("Model", style="cyan", min_width=25)
    for stage in STAGES:
        table.add_column(stage.upper(), justify="center", min_width=10)

    for model in MODELS:
        row = [model]
        for stage in STAGES:
            key = (model, stage)
            if key in results:
                r = results[key]
                row.append(f"{r['rate']:.0%} ({r['successes']}/{r['total']})")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)

    # Print improvement analysis
    console.print("\n[bold]Stage-over-Stage Improvement:[/]")
    for model in MODELS:
        improvements = []
        for i in range(1, len(STAGES)):
            prev = (model, STAGES[i - 1])
            curr = (model, STAGES[i])
            if prev in results and curr in results:
                delta = results[curr]["rate"] - results[prev]["rate"]
                improvements.append(
                    f"  {STAGES[i-1]}→{STAGES[i]}: {delta:+.1%}"
                )
        if improvements:
            console.print(f"\n  [cyan]{model}[/]")
            for imp in improvements:
                console.print(imp)


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument("--results-dir", type=str, default="./outputs")
    args = parser.parse_args()

    compare(args.results_dir)


if __name__ == "__main__":
    main()
