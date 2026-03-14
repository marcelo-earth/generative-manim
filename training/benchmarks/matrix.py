"""Run a benchmark matrix from a manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def _load_manifest(path: str | Path) -> dict:
    manifest_path = _resolve_path(str(path))
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(manifest_path) as f:
        return json.load(f)


def _format_pass_k(value: str | list[int]) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _run_command(command: list[str], dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"$ {rendered}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=TRAINING_ROOT)


def run_matrix(
    manifest_path: str | Path,
    dry_run: bool = False,
    only: str | None = None,
    compare_after: bool = True,
) -> None:
    manifest = _load_manifest(manifest_path)
    manifest_name = Path(manifest_path).stem
    suite = manifest.get("suite", "benchmarks/tasks/core_v1.jsonl")
    suite_path = _resolve_path(suite)
    if suite_path is None:
        raise ValueError("suite is required in manifest")

    output_dir = _resolve_path(
        manifest.get("output_dir", f"./outputs/benchmarks/{manifest_name}")
    )
    assert output_dir is not None
    prompts_path = output_dir / f"{suite_path.stem}_prompts.jsonl"
    samples_per_prompt = int(manifest.get("samples_per_prompt", 5))
    temperature = float(manifest.get("temperature", 0.8))
    seed = manifest.get("seed")
    pass_k = _format_pass_k(manifest.get("pass_k", [1, 5]))

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    _run_command(
        [
            sys.executable,
            "-m",
            "benchmarks.run",
            "export",
            "--suite",
            str(suite_path.relative_to(TRAINING_ROOT)),
            "--output",
            str(prompts_path),
        ],
        dry_run=dry_run,
    )

    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError("manifest must define at least one run")

    selected_runs = []
    for run in runs:
        run_name = run["run_name"]
        model_name = run["model"]
        if only and only not in {run_name, model_name}:
            continue
        selected_runs.append(run)

    if only and not selected_runs:
        raise ValueError(f"No runs matched --only={only}")

    for run in selected_runs:
        model_name = run["model"]
        run_name = run["run_name"]
        run_output_dir = output_dir / model_name / run_name
        if not dry_run:
            run_output_dir.mkdir(parents=True, exist_ok=True)
        responses_path = _resolve_path(run.get("responses_path"))
        checkpoint_path = _resolve_path(run.get("checkpoint"))

        if responses_path is None:
            if checkpoint_path is None:
                raise ValueError(
                    f"Run {run_name} must define either checkpoint or responses_path"
                )
            generate_command = [
                sys.executable,
                "-m",
                "eval.generate_responses",
                "--model",
                model_name,
                "--checkpoint",
                str(checkpoint_path),
                "--test-path",
                str(prompts_path),
                "--output",
                str(run_output_dir / "responses.jsonl"),
                "--temperature",
                str(run.get("temperature", temperature)),
                "--samples-per-prompt",
                str(run.get("samples_per_prompt", samples_per_prompt)),
            ]
            effective_seed = run.get("seed", seed)
            if effective_seed is not None:
                generate_command.extend(["--seed", str(effective_seed)])
            _run_command(generate_command, dry_run=dry_run)
            responses_path = run_output_dir / "responses.jsonl"
        else:
            print(f"Using pre-generated responses for {model_name}/{run_name}: {responses_path}")

        evaluate_command = [
            sys.executable,
            "-m",
            "benchmarks.run",
            "evaluate",
            "--suite",
            str(suite_path.relative_to(TRAINING_ROOT)),
            "--responses",
            str(responses_path),
            "--output-dir",
            str(run_output_dir),
            "--model-name",
            model_name,
            "--run-name",
            run_name,
            "--pass-k",
            _format_pass_k(run.get("pass_k", pass_k)),
        ]
        _run_command(evaluate_command, dry_run=dry_run)

    if compare_after:
        compare_command = [
            sys.executable,
            "-m",
            "benchmarks.compare",
            "--results-dir",
            str(output_dir),
            "--suite",
            suite_path.stem,
        ]
        _run_command(compare_command, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix from a manifest")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()

    run_matrix(
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        only=args.only,
        compare_after=not args.no_compare,
    )


if __name__ == "__main__":
    main()
