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
from benchmarks.pass_k import estimate_pass_at_k
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


def _load_responses(path: str | Path) -> dict[str, list[dict]]:
    responses_path = Path(path)
    grouped: dict[str, list[dict]] = defaultdict(list)

    with open(responses_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" in row:
                grouped[f"task_id:{row['task_id']}"].append(row)
            if "prompt" in row:
                grouped[f"prompt:{row['prompt']}"].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("sample_index", 0)))
    return grouped


def _animation_score(task: BenchmarkTask, animation_count: int) -> float:
    if task.min_animation_count <= 0:
        return 1.0
    return min(animation_count / task.min_animation_count, 1.0)


def _task_score(render_success: bool, required_rate: float, animation_score: float) -> float:
    score = (0.7 * float(render_success)) + (0.2 * required_rate) + (0.1 * animation_score)
    return round(score, 4)


def _parse_pass_k(pass_k: str) -> list[int]:
    values = []
    for chunk in pass_k.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("pass_k must contain at least one positive integer")
    if any(value <= 0 for value in values):
        raise ValueError("pass_k values must be positive")
    return sorted(set(values))


def evaluate_suite(
    suite_path: str | Path,
    responses_path: str | Path,
    output_dir: str | Path,
    model_name: str = "unknown",
    run_name: str = "benchmark",
    max_workers: int = 4,
    timeout: int = 120,
    pass_k_values: list[int] | None = None,
) -> tuple[dict, Path]:
    """Evaluate a response file against a frozen benchmark suite."""
    tasks = load_suite(suite_path)
    responses = _load_responses(responses_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pass_k_values = pass_k_values or [1]

    prepared: list[tuple[BenchmarkTask, dict, str, object]] = []
    pending_codes: list[str] = []
    missing_results: list[dict] = []

    for task in tasks:
        task_responses = responses.get(f"task_id:{task.task_id}") or responses.get(
            f"prompt:{task.prompt}"
        )
        if task_responses is None:
            missing_results.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "prompt": task.prompt,
                    "sample_index": None,
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

        for response in task_responses:
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
                "sample_index": response.get("sample_index", 0),
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

    results.sort(key=lambda row: (row["task_id"], row["sample_index"] is None, row["sample_index"] or 0))

    total_tasks = len(tasks)
    sample_results = [row for row in results if row["error_type"] != "missing_response"]
    total_samples = len(sample_results)
    sample_successes = sum(1 for row in sample_results if row["success"])
    missing = sum(1 for row in results if row["error_type"] == "missing_response")
    required_rate = (
        sum(row["required_pattern_rate"] for row in sample_results) / total_samples
        if total_samples
        else 0.0
    )
    animation_rate = (
        sum(row["animation_score"] for row in sample_results) / total_samples
        if total_samples
        else 0.0
    )
    mean_score = (
        sum(row["score"] for row in sample_results) / total_samples if total_samples else 0.0
    )

    grouped_results: dict[str, list[dict]] = defaultdict(list)
    for row in sample_results:
        grouped_results[row["task_id"]].append(row)

    task_summaries = []
    for task in tasks:
        task_rows = grouped_results.get(task.task_id, [])
        successes = sum(1 for row in task_rows if row["success"])
        task_pass_at_k = {}
        for k in pass_k_values:
            value = estimate_pass_at_k(len(task_rows), successes, k)
            task_pass_at_k[str(k)] = 0.0 if value is None else round(value, 4)

        task_summaries.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "difficulty": task.difficulty,
                "samples": len(task_rows),
                "successes": successes,
                "task_pass": successes > 0,
                "best_score": round(max((row["score"] for row in task_rows), default=0.0), 4),
                "mean_score": round(
                    sum(row["score"] for row in task_rows) / len(task_rows), 4
                )
                if task_rows
                else 0.0,
                "pass_at_k": task_pass_at_k,
            }
        )

    task_pass_rate = (
        sum(1 for row in task_summaries if row["task_pass"]) / total_tasks if total_tasks else 0.0
    )
    best_of_n_score = (
        sum(row["best_score"] for row in task_summaries) / total_tasks if total_tasks else 0.0
    )

    aggregate_pass_at_k = {}
    for k in pass_k_values:
        eligible = [row for row in task_summaries if row["samples"] >= k]
        value = sum(row["pass_at_k"][str(k)] for row in task_summaries) / total_tasks if total_tasks else 0.0
        aggregate_pass_at_k[str(k)] = {
            "value": round(value, 4),
            "eligible_tasks": len(eligible),
        }

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in task_summaries:
        by_category[row["category"]].append(row)

    category_summary = {}
    for category, items in sorted(by_category.items()):
        category_summary[category] = {
            "tasks": len(items),
            "task_pass_rate": round(sum(1 for row in items if row["task_pass"]) / len(items), 4),
            "best_of_n_score": round(sum(row["best_score"] for row in items) / len(items), 4),
        }

    summary = {
        "suite": Path(suite_path).stem,
        "model_name": model_name,
        "run_name": run_name,
        "responses_path": str(Path(responses_path).resolve()),
        "total_tasks": total_tasks,
        "total_samples": total_samples,
        "sample_render_success_rate": round(
            sample_successes / total_samples if total_samples else 0.0, 4
        ),
        "task_pass_rate": round(task_pass_rate, 4),
        "pass_at_k": aggregate_pass_at_k,
        "mean_required_pattern_rate": round(required_rate, 4),
        "mean_animation_score": round(animation_rate, 4),
        "mean_sample_score": round(mean_score, 4),
        "mean_best_of_n_score": round(best_of_n_score, 4),
        "missing_tasks": missing,
        "categories": category_summary,
    }

    results_path = output / f"{model_name}_{run_name}_{summary['suite']}_results.jsonl"
    summary_path = output / f"{model_name}_{run_name}_{summary['suite']}_summary.json"
    tasks_path = output / f"{model_name}_{run_name}_{summary['suite']}_tasks.jsonl"

    with open(results_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    with open(tasks_path, "w") as f:
        for row in task_summaries:
            f.write(json.dumps(row) + "\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    console.print(f"[green]Wrote detailed results to {results_path.resolve()}[/]")
    console.print(f"[green]Wrote task summaries to {tasks_path.resolve()}[/]")
    console.print(f"[green]Wrote summary to {summary_path.resolve()}[/]")
    return summary, results_path


def _print_summary(summary: dict) -> None:
    table = Table(title=f"Benchmark Summary: {summary['model_name']} / {summary['run_name']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Suite", summary["suite"])
    table.add_row("Total Tasks", str(summary["total_tasks"]))
    table.add_row("Total Samples", str(summary["total_samples"]))
    table.add_row("Sample Render Success", f"{summary['sample_render_success_rate']:.1%}")
    table.add_row("Task Pass Rate", f"{summary['task_pass_rate']:.1%}")
    table.add_row(
        "Required Pattern Rate", f"{summary['mean_required_pattern_rate']:.1%}"
    )
    table.add_row("Animation Score", f"{summary['mean_animation_score']:.1%}")
    table.add_row("Mean Sample Score", f"{summary['mean_sample_score']:.1%}")
    table.add_row("Mean Best-of-N Score", f"{summary['mean_best_of_n_score']:.1%}")
    table.add_row("Missing Tasks", str(summary["missing_tasks"]))
    for k, details in summary["pass_at_k"].items():
        value = details["value"]
        label = "n/a" if value is None else f"{value:.1%}"
        table.add_row(f"pass@{k}", f"{label} ({details['eligible_tasks']} tasks)")
    console.print(table)

    category_table = Table(title="Category Breakdown")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Tasks", justify="right")
    category_table.add_column("Task Pass Rate", justify="right")
    category_table.add_column("Best-of-N Score", justify="right")

    for category, details in summary["categories"].items():
        category_table.add_row(
            category,
            str(details["tasks"]),
            f"{details['task_pass_rate']:.1%}",
            f"{details['best_of_n_score']:.1%}",
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
    evaluate_parser.add_argument("--pass-k", type=str, default="1,5")

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
        pass_k_values=_parse_pass_k(args.pass_k),
    )


if __name__ == "__main__":
    main()
