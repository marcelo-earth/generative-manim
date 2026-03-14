#!/bin/bash
# Run the frozen Generative Manim benchmark for a checkpoint.
set -e

MODEL=${1:?Usage: run_benchmark.sh <model> <checkpoint> [suite] [run-name]}
CHECKPOINT=${2:?Usage: run_benchmark.sh <model> <checkpoint> [suite] [run-name]}
SUITE=${3:-benchmarks/tasks/core_v1.jsonl}
RUN_NAME=${4:-benchmark}

cd "$(dirname "$0")/.."

OUTPUT_DIR="${CHECKPOINT}/benchmarks"
PROMPTS_PATH="${OUTPUT_DIR}/$(basename "${SUITE%.jsonl}")_prompts.jsonl"
RESPONSES_PATH="${OUTPUT_DIR}/$(basename "${SUITE%.jsonl}")_responses.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "=== Benchmark: $MODEL / $RUN_NAME ==="
echo ""
echo "Step 1: Export frozen prompts"
python -m benchmarks.run export --suite "$SUITE" --output "$PROMPTS_PATH"

echo ""
echo "Step 2: Generate benchmark responses"
python -m eval.generate_responses \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --test-path "$PROMPTS_PATH" \
  --output "$RESPONSES_PATH"

echo ""
echo "Step 3: Evaluate benchmark results"
python -m benchmarks.run evaluate \
  --suite "$SUITE" \
  --responses "$RESPONSES_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL" \
  --run-name "$RUN_NAME"

echo ""
echo "=== Benchmark Complete ==="
