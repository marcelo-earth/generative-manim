"""Generate model responses on a test set for evaluation."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from ..utils.system_prompt import SYSTEM_PROMPT
from ..utils.config_loader import load_config

DEFAULT_TEST = Path(__file__).parent.parent / "data" / "outputs" / "test_prompts.jsonl"


def load_model(model_name: str, checkpoint: str):
    """Load a fine-tuned model checkpoint."""
    config = load_config(model_name=model_name)

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

    model = PeftModel.from_pretrained(base_model, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_responses(
    model_name: str,
    checkpoint: str,
    test_path: str | Path = DEFAULT_TEST,
    output_path: str | Path | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
):
    """Generate responses for all test prompts."""
    test_path = Path(test_path)

    if output_path is None:
        output_path = Path(checkpoint) / "test_responses.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load test prompts
    prompts = []
    with open(test_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])

    print(f"Loading model: {model_name} from {checkpoint}")
    model, tokenizer = load_model(model_name, checkpoint)
    model.eval()

    print(f"Generating responses for {len(prompts)} test prompts...")

    with open(output_path, "w") as f:
        for prompt in tqdm(prompts, desc="Generating"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

    print(f"Saved responses to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate test responses")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    generate_responses(
        model_name=args.model,
        checkpoint=args.checkpoint,
        test_path=args.test_path,
        output_path=args.output,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
