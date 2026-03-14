"""Compare benchmark summaries across models and runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _load_summaries(results_dir: str | Path, suite: str | None = None) -> list[dict]:
    results_root = Path(results_dir)
    summaries: list[dict] = []

    for path in sorted(results_root.rglob("*_summary.json")):
        with open(path) as f:
            data = json.load(f)
        if suite and data.get("suite") != suite:
            continue
        data["_path"] = str(path.resolve())
        summaries.append(data)

    return summaries


def _sort_key(summary: dict) -> tuple:
    pass_at_k = summary.get("pass_at_k", {})
    pass_at_5 = (pass_at_k.get("5") or {}).get("value")
    pass_at_1 = (pass_at_k.get("1") or {}).get("value")
    return (
        1 if summary.get("total_samples", 0) > 0 else 0,
        pass_at_5 if pass_at_5 is not None else -1.0,
        pass_at_1 if pass_at_1 is not None else -1.0,
        summary.get("task_pass_rate", -1.0),
        summary.get("mean_best_of_n_score", -1.0),
    )


def compare(
    results_dir: str | Path = "./outputs/benchmarks",
    suite: str | None = None,
    limit: int | None = None,
    output_csv: str | Path | None = None,
) -> list[dict]:
    """Load and compare benchmark summaries."""
    summaries = _load_summaries(results_dir, suite=suite)
    if not summaries:
        console.print("[red]No benchmark summaries found.[/]")
        console.print(f"Looking in: {Path(results_dir).resolve()}")
        if suite:
            console.print(f"Suite filter: {suite}")
        return []

    summaries.sort(key=_sort_key, reverse=True)
    if limit is not None:
        summaries = summaries[:limit]

    _print_table(summaries)

    if output_csv is not None:
        _write_csv(summaries, output_csv)

    return summaries


def _print_table(summaries: list[dict]) -> None:
    table = Table(title="Benchmark Leaderboard")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Model", style="cyan", min_width=20)
    table.add_column("Run", min_width=12)
    table.add_column("Suite", min_width=10)
    table.add_column("pass@1", justify="right")
    table.add_column("pass@5", justify="right")
    table.add_column("Task Pass", justify="right")
    table.add_column("Best-of-N", justify="right")
    table.add_column("Samples", justify="right")

    for index, summary in enumerate(summaries, start=1):
        pass_at_k = summary.get("pass_at_k", {})
        pass_at_1 = (pass_at_k.get("1") or {}).get("value")
        pass_at_5 = (pass_at_k.get("5") or {}).get("value")
        table.add_row(
            str(index),
            summary.get("model_name", "unknown"),
            summary.get("run_name", "unknown"),
            summary.get("suite", "unknown"),
            _fmt_pct(pass_at_1),
            _fmt_pct(pass_at_5),
            _fmt_pct(summary.get("task_pass_rate")),
            _fmt_pct(summary.get("mean_best_of_n_score")),
            str(summary.get("total_samples", 0)),
        )

    console.print(table)

    best = summaries[0]
    console.print(
        f"[bold]Top run:[/] {best.get('model_name')} / {best.get('run_name')} "
        f"(pass@1={_fmt_pct((best.get('pass_at_k', {}).get('1') or {}).get('value'))}, "
        f"pass@5={_fmt_pct((best.get('pass_at_k', {}).get('5') or {}).get('value'))})"
    )


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _write_csv(summaries: list[dict], output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "run_name",
                "suite",
                "total_tasks",
                "total_samples",
                "task_pass_rate",
                "pass_at_1",
                "pass_at_5",
                "mean_best_of_n_score",
                "mean_sample_score",
                "summary_path",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            pass_at_k = summary.get("pass_at_k", {})
            writer.writerow(
                {
                    "model_name": summary.get("model_name"),
                    "run_name": summary.get("run_name"),
                    "suite": summary.get("suite"),
                    "total_tasks": summary.get("total_tasks"),
                    "total_samples": summary.get("total_samples"),
                    "task_pass_rate": summary.get("task_pass_rate"),
                    "pass_at_1": (pass_at_k.get("1") or {}).get("value"),
                    "pass_at_5": (pass_at_k.get("5") or {}).get("value"),
                    "mean_best_of_n_score": summary.get("mean_best_of_n_score"),
                    "mean_sample_score": summary.get("mean_sample_score"),
                    "summary_path": summary.get("_path"),
                }
            )

    console.print(f"[green]Wrote CSV leaderboard to {output_path.resolve()}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark summaries")
    parser.add_argument("--results-dir", type=str, default="./outputs/benchmarks")
    parser.add_argument("--suite", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    compare(
        results_dir=args.results_dir,
        suite=args.suite,
        limit=args.limit,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
