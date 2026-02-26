#!/bin/bash
# Run DPO training: generate pairs + train
set -e

MODEL=${1:?Usage: run_dpo.sh <model> <sft-checkpoint>}
SFT_CHECKPOINT=${2:?Usage: run_dpo.sh <model> <sft-checkpoint>}

echo "=== DPO Pipeline: $MODEL ==="
cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Generate DPO pairs from SFT model"
python -m data.generate_dpo_pairs --model "$MODEL" --sft-checkpoint "$SFT_CHECKPOINT"

echo ""
echo "Step 2: DPO training"
python -m stages.dpo_train --model "$MODEL" --sft-checkpoint "$SFT_CHECKPOINT" "${@:3}"

echo ""
echo "=== DPO Pipeline Complete ==="
