"""Generate DPO preference pairs: SFT model → 4 outputs/prompt → best/worst by render success."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from ..rendering.manim_verifier import verify_code
from ..utils.code_extraction import clean_code
from ..utils.system_prompt import SYSTEM_PROMPT
from ..utils.config_loader import load_config

DEFAULT_INPUT = Path(__file__).parent / "outputs" / "sft_train.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "dpo_train.jsonl"


def load_sft_model(model_name: str, sft_checkpoint: str):
    """Load SFT checkpoint with QLoRA."""
    config = load_config(model_name=model_name, stage="sft")

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

    model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_outputs(
    model,
    tokenizer,
    prompt: str,
    n: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
) -> list[str]:
    """Generate n different completions for a prompt."""
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
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        decoded = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        completions.append(decoded)

    return completions


def generate_dpo_pairs(
    model_name: str,
    sft_checkpoint: str,
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    pairs_per_prompt: int = 4,
    limit: int | None = None,
    timeout: int = 120,
):
    """Generate DPO pairs from SFT model outputs."""
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

    print(f"Loading SFT model: {model_name} from {sft_checkpoint}")
    model, tokenizer = load_sft_model(model_name, sft_checkpoint)
    model.eval()

    print(f"Generating DPO pairs for {len(prompts)} prompts...")
    total_pairs = 0

    with open(output_path, "w") as f:
        for i, prompt in enumerate(prompts):
            # Generate multiple outputs
            completions = generate_outputs(
                model, tokenizer, prompt, n=pairs_per_prompt
            )

            # Score each by rendering
            scored = []
            for comp in completions:
                code = clean_code(comp)
                result = verify_code(code, timeout=timeout)
                scored.append((comp, code, result))

            # Find best (renders) and worst (doesn't render)
            renders = [(comp, code, r) for comp, code, r in scored if r.success]
            fails = [(comp, code, r) for comp, code, r in scored if not r.success]

            if renders and fails:
                # Pick best render (most animations) and worst fail
                chosen = max(renders, key=lambda x: x[2].animation_count)
                rejected = fails[0]

                pair = {
                    "prompt": prompt,
                    "chosen": chosen[0],
                    "rejected": rejected[0],
                    "chosen_renders": True,
                    "rejected_renders": False,
                }
                f.write(json.dumps(pair) + "\n")
                total_pairs += 1

            if (i + 1) % 10 == 0:
                render_rate = sum(1 for _, _, r in scored if r.success) / len(scored) * 100
                print(f"  [{i+1}/{len(prompts)}] Pairs: {total_pairs} | Batch render rate: {render_rate:.0f}%")

    print(f"\nDone. Generated {total_pairs} DPO pairs.")


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sft-checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pairs-per-prompt", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    generate_dpo_pairs(
        model_name=args.model,
        sft_checkpoint=args.sft_checkpoint,
        input_path=args.input,
        output_path=args.output,
        pairs_per_prompt=args.pairs_per_prompt,
        limit=args.limit,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
