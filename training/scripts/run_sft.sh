#!/bin/bash
# Run SFT training for a specified model
set -e

MODEL=${1:-qwen2.5-coder-7b}
echo "=== SFT Training: $MODEL ==="
cd "$(dirname "$0")/.."

python -m stages.sft_train --model "$MODEL" "${@:2}"
