#!/bin/bash
# Run evaluation for a model checkpoint
set -e

MODEL=${1:?Usage: run_eval.sh <model> <checkpoint> <stage>}
CHECKPOINT=${2:?Usage: run_eval.sh <model> <checkpoint> <stage>}
STAGE=${3:?Usage: run_eval.sh <model> <checkpoint> <stage>}

echo "=== Evaluation: $MODEL / $STAGE ==="
cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Generate test responses"
python -m eval.generate_responses --model "$MODEL" --checkpoint "$CHECKPOINT"

echo ""
echo "Step 2: Evaluate responses"
RESPONSES="${CHECKPOINT}/test_responses.jsonl"
python -m eval.evaluate --responses "$RESPONSES" --model-name "$MODEL" --stage "$STAGE"

echo ""
echo "=== Evaluation Complete ==="
