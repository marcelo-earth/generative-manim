# Generative Manim with Featherless

Generative Manim can use [Featherless](https://featherless.ai) as an OpenAI-compatible provider for open-weight models. This is useful for two workflows:

1. Generate Manim code from prompts through the API.
2. Benchmark hosted open models with the render-based Manim verifier.

## API Setup

Create a Featherless API key from your account dashboard, then export it:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
```

Run the API:

```bash
python run.py
```

Generate Manim code with a Featherless model:

```bash
curl -X POST http://127.0.0.1:8080/v1/code/generation \
  -H "Content-Type: application/json" \
  -d '{
    "engine": "featherless",
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "prompt": "Create a Manim animation that explains the Pythagorean theorem with a right triangle and colored squares."
  }'
```

The chat endpoint also accepts `engine: "featherless"`:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/generation \
  -H "Content-Type: application/json" \
  -d '{
    "engine": "featherless",
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Create a Manim scene that animates a sine wave being drawn over axes."
      }
    ]
  }'
```

## Benchmark Hosted Models

The benchmark path evaluates generated code by rendering it with Manim, then reporting render success, required Manim patterns, animation-count compliance, and pass@k.

Set your key:

```bash
export FEATHERLESS_API_KEY="your-featherless-key"
```

Dry-run the Featherless benchmark matrix:

```bash
cd training
python -m benchmarks.matrix \
  --manifest benchmarks/manifests/featherless_core_v1.json \
  --dry-run
```

Run one model first:

```bash
cd training
python -m benchmarks.matrix \
  --manifest benchmarks/manifests/featherless_core_v1.json \
  --only qwen2.5-coder-7b-instruct-featherless
```

Run the full matrix:

```bash
cd training
python -m benchmarks.matrix \
  --manifest benchmarks/manifests/featherless_core_v1.json
```

Results are written to:

```text
training/outputs/benchmarks/featherless_core_v1/
```

## Lower-Cost Smoke Test

To conserve tokens, generate responses for only a few benchmark prompts:

```bash
cd training
python -m benchmarks.run export \
  --suite benchmarks/tasks/core_v1.jsonl \
  --output ./outputs/benchmarks/featherless_smoke_prompts.jsonl

python -m eval.generate_remote_responses \
  --provider featherless \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --test-path ./outputs/benchmarks/featherless_smoke_prompts.jsonl \
  --output ./outputs/benchmarks/featherless_smoke_responses.jsonl \
  --samples-per-prompt 1 \
  --temperature 0.2 \
  --limit 3
```

Then evaluate those responses:

```bash
python -m benchmarks.run evaluate \
  --suite benchmarks/tasks/core_v1.jsonl \
  --responses ./outputs/benchmarks/featherless_smoke_responses.jsonl \
  --output-dir ./outputs/benchmarks/featherless_smoke \
  --model-name qwen2.5-coder-7b-instruct-featherless \
  --run-name smoke \
  --pass-k 1
```

Partial smoke runs will show missing tasks in the summary. Full benchmark runs should use the matrix manifest.

## Demo Prompts

Good demo prompts for visual examples:

- `Create a Manim animation that explains the Pythagorean theorem with a right triangle and colored squares.`
- `Create a scene where a sine wave is drawn over axes while a dot moves along the curve.`
- `Animate gradient descent on a simple parabola, showing the point stepping toward the minimum.`

## Notes

- Featherless is OpenAI-compatible, so the integration uses the OpenAI Python SDK with `base_url="https://api.featherless.ai/v1"`.
- Hosted open models vary in instruction following. The benchmark is designed to make that variation visible through render success and pass@k.
- For public results, prefer `samples_per_prompt >= 3` so pass@k is meaningful.
