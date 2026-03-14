# Generative Manim Benchmark

This folder contains the first frozen benchmark suite for `generative-manim`.

## What a programming benchmark should do

For programming models, the benchmark should not be a vague gallery of prompts. It needs:

1. A **frozen task suite** so runs are comparable over time.
2. An **execution-based primary metric** so models are graded on runnable outputs, not style.
3. **Domain-specific checks** so "technically runnable but wrong for the task" does not look good.
4. A **reproducible report format** so results can be versioned, compared, and published.

For Manim, the primary benchmark question is:

> Given a natural-language animation request, can the model generate code that both renders and uses the right Manim concepts?

That means the benchmark starts with render success, then adds domain checks like:

- Does the scene use `Axes`, `Surface`, `ValueTracker`, `MathTex`, or camera controls when the prompt requires them?
- Does it produce enough animation steps to count as an explanation rather than a static render?
- Does it avoid failing to render entirely?

## Current MVP

The current suite is `tasks/core_v1.jsonl`.

Each task contains:

- `task_id`
- `category`
- `difficulty`
- `prompt`
- `required_patterns`
- `disallowed_patterns`
- `min_animation_count`

The sample-level scoring model is intentionally simple and transparent:

- `70%` render success
- `20%` required-pattern coverage
- `10%` animation-count compliance

For stochastic models, the benchmark also reports task-level `pass@k`.

That matters because programming models are not judged well by a single sample. A strong model may fail once and succeed on the second or third try. `pass@k` measures the probability that at least one of `k` samples solves the task, using the standard code-benchmark estimator.

This is a good starting point for expert-programming evaluation because it is:

- objective
- cheap to run
- aligned with the repository's existing Manim verifier

It is not the final form. The next steps should be:

1. Add prompt-specific semantic checks beyond regex.
2. Add visual regression or reference-image scoring for tasks with clear expected layouts.
3. Add pass@k evaluation for stochastic models.
4. Split suites into `core`, `advanced`, and `research` tracks.

## Workflow

Export the frozen suite into prompt JSONL:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/core_v1_prompts.jsonl
```

Generate multiple samples per prompt for pass@k:

```bash
python -m eval.generate_responses \
  --model qwen2.5-coder-7b \
  --checkpoint ./outputs/grpo/qwen2.5-coder-7b \
  --test-path ./outputs/benchmarks/core_v1_prompts.jsonl \
  --output ./outputs/benchmarks/qwen_core_v1_responses.jsonl \
  --temperature 0.8 \
  --samples-per-prompt 5 \
  --seed 42
```

Evaluate the responses:

```bash
python -m benchmarks.run evaluate \
  --suite benchmarks/tasks/core_v1.jsonl \
  --responses ./outputs/benchmarks/qwen_core_v1_responses.jsonl \
  --output-dir ./outputs/benchmarks \
  --model-name qwen2.5-coder-7b \
  --run-name grpo \
  --pass-k 1,5
```

If you request `pass@5`, make sure you generated at least `5` samples per task. The summary also reports how many tasks were actually eligible for each `k`.

## Result files

Evaluation writes:

- `*_results.jsonl`: per-task outcomes
- `*_tasks.jsonl`: per-task aggregated summaries, including pass@k
- `*_summary.json`: aggregate metrics and category breakdown

Compare benchmark runs across models:

```bash
python -m benchmarks.compare \
  --results-dir ./outputs/benchmarks \
  --suite core_v1
```

Export a CSV leaderboard:

```bash
python -m benchmarks.compare \
  --results-dir ./outputs/benchmarks \
  --suite core_v1 \
  --output-csv ./outputs/benchmarks/core_v1_leaderboard.csv
```

Run a standardized benchmark matrix from a manifest:

```bash
python -m benchmarks.matrix \
  --manifest benchmarks/manifests/open_source_core_v1.json \
  --dry-run
```

The manifest can mix:

- local checkpoints via `checkpoint`
- pre-generated API outputs via `responses_path`

That lets the same benchmark stack compare open-source checkpoints and hosted model baselines.
