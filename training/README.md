# Generative Manim — Open-Source Training Pipeline

Train specialized open-source models to generate Manim animation code using a 3-stage pipeline: **SFT → DPO → GRPO**.

The key insight: the Manim renderer is a **deterministic verifier** — code either renders or it doesn't. This replaces the need for a reward model in GRPO, exactly like DeepSeek-R1 uses math answer checkers.

## Base Models

| Model | HF ID | Parameters |
|-------|--------|-----------|
| Qwen 2.5 Coder | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B |
| DeepSeek Coder V2 Lite | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B (2.4B active) |
| CodeLlama | `codellama/CodeLlama-7b-Instruct-hf` | 7B |

All models use the same QLoRA recipe: **4-bit NF4, rank=32, alpha=64, target all-linear layers**.

## Pipeline Overview

```
Seeds (75) → GPT-4o Expansion (8K prompts) → Teacher Completions → Manim Validation
                                                                         ↓
                                                               Validated Dataset (5K+)
                                                                         ↓
                                                            ┌────────────┼────────────┐
                                                            ↓            ↓            ↓
                                                     SFT Training  DPO Training  GRPO Training
                                                     (3 epochs)    (1 epoch)     (1 epoch)
                                                            ↓            ↓            ↓
                                                         Eval →      Eval →      Eval → Compare
```

### Stage 1: SFT (Supervised Fine-Tuning)
- Train on validated prompt→code pairs
- Uses TRL `SFTTrainer` with QLoRA
- 3 epochs, lr=2e-4, cosine schedule

### Stage 2: DPO (Direct Preference Optimization)
- Generate 4 outputs per prompt from SFT model
- Render each → renders = chosen, fails = rejected
- Train with TRL `DPOTrainer`, beta=0.1

### Stage 3: GRPO (Group Relative Policy Optimization)
- Generate 8 outputs per prompt from DPO model
- Score each with Manim verifier (render_success + animation_bonus)
- Group-relative advantages: `(reward - mean) / std`
- Train with TRL `GRPOTrainer`

## Quick Start

### 1. Install dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Run data pipeline

```bash
# Generate synthetic prompts (requires OPENAI_API_KEY)
python -m data.generate_prompts

# Generate teacher completions
python -m data.generate_completions

# Validate renders (requires manim installed)
python -m data.validate_renders

# Prepare SFT dataset
python -m data.prepare_sft_dataset
```

### 3. Train

```bash
# SFT
python -m stages.sft_train --model qwen2.5-coder-7b

# DPO (after generating pairs)
python -m data.generate_dpo_pairs --model qwen2.5-coder-7b --sft-checkpoint ./outputs/sft/qwen2.5-coder-7b
python -m stages.dpo_train --model qwen2.5-coder-7b --sft-checkpoint ./outputs/sft/qwen2.5-coder-7b

# GRPO
python -m stages.grpo_train --model qwen2.5-coder-7b --dpo-checkpoint ./outputs/dpo/qwen2.5-coder-7b
```

### 4. Evaluate

```bash
python -m eval.generate_responses --model qwen2.5-coder-7b --checkpoint ./outputs/grpo/qwen2.5-coder-7b
python -m eval.evaluate --responses ./outputs/grpo/qwen2.5-coder-7b/test_responses.jsonl --model-name qwen2.5-coder-7b --stage grpo
python -m eval.compare_models --results-dir ./outputs
```

## Kaggle Notebooks

Self-contained notebooks that can be uploaded directly to Kaggle:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Explore datasets and distributions |
| `02_sft_kaggle.ipynb` | SFT training on Kaggle T4 |
| `03_dpo_kaggle.ipynb` | DPO training on Kaggle T4 |
| `04_grpo_kaggle.ipynb` | GRPO with live Manim rendering |
| `05_evaluation.ipynb` | Cross-model comparison |

## The Manim Verifier

The verifier (`rendering/manim_verifier.py`) replicates the rendering pattern from the GM API:

1. Creates isolated temp directory
2. Writes code to `scene.py` with required imports
3. Extracts class name via regex
4. Runs: `manim scene.py GenScene --format=mp4 -ql`
5. Timeout: 120s (60s on Kaggle)
6. Returns: `VerifyResult(success, error_type, animation_count, render_time)`

**Reward function** (`rendering/reward.py`):
- `render_success`: 1.0 if renders, 0.0 if fails
- `animation_bonus`: 0.1 per `self.play()` call (capped at 0.5)
- Total range: 0.0 to 1.5 — deterministic, no reward model needed

## Directory Structure

```
training/
├── config/          # YAML configs (base, models, stages)
├── data/            # Data generation pipeline
├── rendering/       # Manim verifier + reward function
├── stages/          # SFT, DPO, GRPO training scripts
├── eval/            # Evaluation harness
├── inference/       # vLLM server + LocalModelEngine
├── utils/           # Config loader, logging, code extraction
├── notebooks/       # Kaggle-ready Jupyter notebooks
└── scripts/         # Shell scripts for convenience
```

## Budget

| Item | Cost |
|------|------|
| Prompt generation (GPT-4o) | ~$3 |
| Teacher completions (GPT-4o) | ~$20-25 |
| Misc API calls | ~$5 |
| Training compute (Kaggle T4) | $0 |
| **Total** | **~$30-35** |
