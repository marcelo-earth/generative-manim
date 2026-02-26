"""Generate GRPO rollouts: DPO model → 8 outputs/prompt → score with Manim verifier."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from ..rendering.reward import compute_reward
from ..utils.code_extraction import clean_code
from ..utils.system_prompt import SYSTEM_PROMPT
from ..utils.config_loader import load_config

DEFAULT_INPUT = Path(__file__).parent / "outputs" / "sft_train.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "grpo_rollouts.jsonl"


def load_dpo_model(model_name: str, dpo_checkpoint: str):
    """Load DPO checkpoint with QLoRA."""
    config = load_config(model_name=model_name, stage="dpo")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.model.get("trust_remote_code", False),
    )

    model = PeftModel.from_pretrained(base_model, dpo_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_rollouts(
    model,
    tokenizer,
    prompt: str,
    group_size: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
) -> list[str]:
    """Generate group_size completions for GRPO."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        decoded = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        completions.append(decoded)

    return completions


def generate_all_rollouts(
    model_name: str,
    dpo_checkpoint: str,
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    group_size: int = 8,
    limit: int | None = None,
    timeout: int = 120,
):
    """Pre-generate and score rollouts for offline GRPO."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            prompt = data["messages"][1]["content"]
            prompts.append(prompt)

    if limit:
        prompts = prompts[:limit]

    print(f"Loading DPO model: {model_name} from {dpo_checkpoint}")
    model, tokenizer = load_dpo_model(model_name, dpo_checkpoint)
    model.eval()

    print(f"Generating GRPO rollouts for {len(prompts)} prompts (group_size={group_size})...")

    with open(output_path, "w") as f:
        for i, prompt in enumerate(prompts):
            completions = generate_rollouts(model, tokenizer, prompt, group_size=group_size)

            # Score each completion
            scored = []
            for comp in completions:
                code = clean_code(comp)
                reward, result = compute_reward(code, timeout=timeout)
                scored.append({
                    "completion": comp,
                    "code": code,
                    "reward": reward,
                    "renders": result.success,
                    "animation_count": result.animation_count,
                    "error_type": result.error_type.value if not result.success else None,
                })

            rollout = {
                "prompt": prompt,
                "rollouts": scored,
                "mean_reward": sum(s["reward"] for s in scored) / len(scored),
                "render_rate": sum(1 for s in scored if s["renders"]) / len(scored),
            }
            f.write(json.dumps(rollout) + "\n")

            if (i + 1) % 10 == 0:
                print(
                    f"  [{i+1}/{len(prompts)}] "
                    f"Render rate: {rollout['render_rate']:.0%} | "
                    f"Mean reward: {rollout['mean_reward']:.2f}"
                )

    print(f"\nDone. Generated rollouts for {len(prompts)} prompts.")


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO rollouts")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dpo-checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    generate_all_rollouts(
        model_name=args.model,
        dpo_checkpoint=args.dpo_checkpoint,
        input_path=args.input,
        output_path=args.output,
        group_size=args.group_size,
        limit=args.limit,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
