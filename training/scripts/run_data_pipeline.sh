#!/bin/bash
# Run the full data pipeline: prompts → completions → validation → SFT dataset
set -e

echo "=== Generative Manim Data Pipeline ==="
cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Generate prompts (GPT-4o expansion)"
python -m data.generate_prompts "$@"

echo ""
echo "Step 2: Generate completions (teacher model)"
python -m data.generate_completions "$@"

echo ""
echo "Step 3: Validate renders"
python -m data.validate_renders "$@"

echo ""
echo "Step 4: Prepare SFT dataset"
python -m data.prepare_sft_dataset "$@"

echo ""
echo "=== Data Pipeline Complete ==="
