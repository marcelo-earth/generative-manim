#!/bin/bash
# Run GRPO training with Manim verifier reward
set -e

MODEL=${1:?Usage: run_grpo.sh <model> <dpo-checkpoint>}
DPO_CHECKPOINT=${2:?Usage: run_grpo.sh <model> <dpo-checkpoint>}

echo "=== GRPO Pipeline: $MODEL ==="
cd "$(dirname "$0")/.."

echo ""
echo "GRPO training (with live Manim rendering)"
python -m stages.grpo_train --model "$MODEL" --dpo-checkpoint "$DPO_CHECKPOINT" "${@:3}"

echo ""
echo "=== GRPO Pipeline Complete ==="
