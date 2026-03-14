"""Export and evaluate frozen benchmark suites."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmarks import DEFAULT_SUITE
from benchmarks.checks import run_static_checks
from benchmarks.suite import BenchmarkTask, load_suite
from rendering.manim_verifier import batch_verify
from utils.code_extraction import clean_code

console = Console()


def export_prompts(
    suite_path: str | Path = DEFAULT_SUITE,
    output_path: str | Path = "benchmark_prompts.jsonl",
) -> Path:
    """Export benchmark prompts in the same JSONL shape used by response generation."""
    tasks = load_suite(suite_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for task in tasks:
            payload = {
                "task_id": task.task_id,
                "category": task.category,
                "difficulty": task.difficulty,
                "prompt": task.prompt,
            }
            f.write(json.dumps(payload) + "\n")

    console.print(
        f"[green]Exported {len(tasks)} benchmark prompts to {output.resolve()}[/]"
    )
    return output


def _load_responses(path: str | Path) -> dict[str, dict]:
    responses_path = Path(path)
    responses_by_id: dict[str, dict] = {}
    responses_by_prompt: dict[str, dict] = {}

    with open(responses_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" in row:
                responses_by_id[str(row["task_id"])] = row
            if "prompt" in row:
                responses_by_prompt[str(row["prompt"])] = row

    merged = dict(responses_by_prompt)
    merged.update(responses_by_id)
    return merged


def _animation_score(task: BenchmarkTask, animation_count: int) -> float:
    if task.min_animation_count <= 0:
        return 1.0
    return min(animation_count / task.min_animation_count, 1.0)


def _task_score(render_success: bool, required_rate: float, animation_score: float) -> float:
    score = (0.7 * float(render_success)) + (0.2 * required_rate) + (0.1 * animation_score)
    return round(score, 4)


def evaluate_suite(
    suite_path: str | Path,
    responses_path: str | Path,
    output_dir: str | Path,
    model_name: str = "unknown",
    run_name: str = "benchmark",
    max_workers: int = 4,
    timeout: int = 120,
) -> tuple[dict, Path]:
    """Evaluate a response file against a frozen benchmark suite."""
    tasks = load_suite(suite_path)
    responses = _load_responses(responses_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[BenchmarkTask, dict, str, object]] = []
    pending_codes: list[str] = []
    missing_results: list[dict] = []

    for task in tasks:
        response = responses.get(task.task_id) or responses.get(task.prompt)
        if response is None:
            missing_results.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "prompt": task.prompt,
                    "success": False,
                    "error_type": "missing_response",
                    "error_message": "No response found for task",
                    "animation_count": 0,
                    "render_time": 0.0,
                    "required_pattern_rate": 0.0,
                    "matched_patterns": [],
                    "missing_patterns": task.required_patterns,
                    "disallowed_matches": [],
                    "animation_score": 0.0,
                    "score": 0.0,
                }
            )
            continue

        code = clean_code(response["response"])
        static_result = run_static_checks(task, code)
        pending_codes.append(code)
        prepared.append((task, response, code, static_result))

    verify_results = (
        batch_verify(pending_codes, max_workers=max_workers, timeout=timeout)
        if pending_codes
        else []
    )

    results = list(missing_results)

    for (task, response, code, static_result), verify_result in zip(prepared, verify_results):
        animation_score = _animation_score(task, verify_result.animation_count)
        score = _task_score(
            render_success=verify_result.success,
            required_rate=static_result.required_pattern_rate,
            animation_score=animation_score,
        )
        results.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "difficulty": task.difficulty,
                "prompt": task.prompt,
                "success": verify_result.success,
                "error_type": verify_result.error_type.value,
                "error_message": verify_result.error_message,
                "animation_count": verify_result.animation_count,
                "render_time": round(verify_result.render_time, 3),
                "required_pattern_rate": round(static_result.required_pattern_rate, 4),
                "matched_patterns": static_result.matched_patterns,
                "missing_patterns": static_result.missing_patterns,
                "disallowed_matches": static_result.disallowed_matches,
                "animation_score": round(animation_score, 4),
                "score": score,
                "response_task_id": response.get("task_id"),
                "code": code,
            }
        )

    results.sort(key=lambda row: row["task_id"])

    total_tasks = len(tasks)
    successes = sum(1 for row in results if row["success"])
    missing = sum(1 for row in results if row["error_type"] == "missing_response")
    required_rate = (
        sum(row["required_pattern_rate"] for row in results) / total_tasks if total_tasks else 0.0
    )
    animation_rate = (
        sum(row["animation_score"] for row in results) / total_tasks if total_tasks else 0.0
    )
    mean_score = sum(row["score"] for row in results) / total_tasks if total_tasks else 0.0

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)

    category_summary = {}
    for category, items in sorted(by_category.items()):
        category_summary[category] = {
            "tasks": len(items),
            "render_success_rate": round(
                sum(1 for row in items if row["success"]) / len(items), 4
            ),
            "mean_score": round(sum(row["score"] for row in items) / len(items), 4),
        }

    summary = {
        "suite": Path(suite_path).stem,
        "model_name": model_name,
        "run_name": run_name,
        "responses_path": str(Path(responses_path).resolve()),
        "total_tasks": total_tasks,
        "render_success_rate": round(successes / total_tasks if total_tasks else 0.0, 4),
        "mean_required_pattern_rate": round(required_rate, 4),
        "mean_animation_score": round(animation_rate, 4),
        "mean_score": round(mean_score, 4),
        "missing_tasks": missing,
        "categories": category_summary,
    }

    results_path = output / f"{model_name}_{run_name}_{summary['suite']}_results.jsonl"
    summary_path = output / f"{model_name}_{run_name}_{summary['suite']}_summary.json"

    with open(results_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    console.print(f"[green]Wrote detailed results to {results_path.resolve()}[/]")
    console.print(f"[green]Wrote summary to {summary_path.resolve()}[/]")
    return summary, results_path


def _print_summary(summary: dict) -> None:
    table = Table(title=f"Benchmark Summary: {summary['model_name']} / {summary['run_name']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Suite", summary["suite"])
    table.add_row("Total Tasks", str(summary["total_tasks"]))
    table.add_row("Render Success Rate", f"{summary['render_success_rate']:.1%}")
    table.add_row(
        "Required Pattern Rate", f"{summary['mean_required_pattern_rate']:.1%}"
    )
    table.add_row("Animation Score", f"{summary['mean_animation_score']:.1%}")
    table.add_row("Mean Score", f"{summary['mean_score']:.1%}")
    table.add_row("Missing Tasks", str(summary["missing_tasks"]))
    console.print(table)

    category_table = Table(title="Category Breakdown")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Tasks", justify="right")
    category_table.add_column("Render Success", justify="right")
    category_table.add_column("Mean Score", justify="right")

    for category, details in summary["categories"].items():
        category_table.add_row(
            category,
            str(details["tasks"]),
            f"{details['render_success_rate']:.1%}",
            f"{details['mean_score']:.1%}",
        )

    console.print(category_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and evaluate Manim benchmark suites")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export prompts from a frozen suite")
    export_parser.add_argument("--suite", type=str, default=str(DEFAULT_SUITE))
    export_parser.add_argument("--output", type=str, required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model responses against a frozen suite"
    )
    evaluate_parser.add_argument("--suite", type=str, default=str(DEFAULT_SUITE))
    evaluate_parser.add_argument("--responses", type=str, required=True)
    evaluate_parser.add_argument("--output-dir", type=str, default="./outputs/benchmarks")
    evaluate_parser.add_argument("--model-name", type=str, default="unknown")
    evaluate_parser.add_argument("--run-name", type=str, default="benchmark")
    evaluate_parser.add_argument("--workers", type=int, default=4)
    evaluate_parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    if args.command == "export":
        export_prompts(args.suite, args.output)
        return

    evaluate_suite(
        suite_path=args.suite,
        responses_path=args.responses,
        output_dir=args.output_dir,
        model_name=args.model_name,
        run_name=args.run_name,
        max_workers=args.workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
