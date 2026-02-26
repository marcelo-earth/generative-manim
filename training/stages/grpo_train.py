"""GRPO training with TRL GRPOTrainer + Manim verifier as reward function."""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

from ..rendering.reward import compute_reward
from ..utils.code_extraction import clean_code
from ..utils.config_loader import load_config
from ..utils.logging_utils import setup_wandb, finish_wandb


def make_reward_fn(timeout: int = 120, reward_config: dict | None = None):
    """Create a reward function compatible with TRL GRPOTrainer."""
    rc = reward_config or {}
    render_weight = rc.get("render_success", 1.0)
    bonus_per_play = rc.get("animation_bonus_per_play", 0.1)
    bonus_cap = rc.get("animation_bonus_cap", 0.5)

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """Score a batch of completions using the Manim verifier."""
        rewards = []
        for completion in completions:
            code = clean_code(completion)
            reward, _ = compute_reward(
                code,
                render_success_weight=render_weight,
                animation_bonus_per_play=bonus_per_play,
                animation_bonus_cap=bonus_cap,
                timeout=timeout,
            )
            rewards.append(reward)
        return rewards

    return reward_fn


def train(
    model_name: str,
    dpo_checkpoint: str,
    overrides: dict | None = None,
):
    """Run GRPO training with Manim verifier reward."""
    config = load_config(model_name=model_name, stage="grpo", overrides=overrides)
    grpo_cfg = config.grpo
    model_cfg = config.model

    print(f"=== GRPO Training: {model_cfg.name} ===")
    print(f"  Base: {model_cfg.hf_id}")
    print(f"  DPO checkpoint: {dpo_checkpoint}")
    print(f"  Group size: {grpo_cfg.group_size}")
    print(f"  KL coeff: {grpo_cfg.kl_coeff}")

    # Setup W&B
    run_name = f"grpo-{model_cfg.name}"
    setup_wandb(config, run_name=run_name, tags=["grpo", model_cfg.name])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.qlora.load_in_4bit,
        bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
    )

    # Load base model + DPO adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, dpo_checkpoint, is_trainable=True)

    # LoRA config for GRPO
    lora_cfg = config.lora
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    # Load prompts dataset
    prompt_path = config.paths.get("grpo_prompts", config.paths.sft_train)
    dataset = load_dataset("json", data_files=prompt_path, split="train")

    # Extract just prompts
    def extract_prompt(example):
        if "messages" in example:
            for msg in example["messages"]:
                if msg["role"] == "user":
                    return {"prompt": msg["content"]}
        return {"prompt": example.get("prompt", "")}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    print(f"  Prompts: {len(dataset)}")

    # Create reward function
    reward_config = dict(config.get("reward", {}))
    reward_fn = make_reward_fn(
        timeout=config.rendering.timeout,
        reward_config=reward_config,
    )

    # Output directory
    output_dir = os.path.join(grpo_cfg.output_dir, model_cfg.name)

    # GRPO config
    grpo_training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=grpo_cfg.num_train_epochs,
        learning_rate=grpo_cfg.learning_rate,
        lr_scheduler_type=grpo_cfg.lr_scheduler_type,
        warmup_ratio=grpo_cfg.warmup_ratio,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        fp16=grpo_cfg.fp16,
        bf16=grpo_cfg.bf16,
        logging_steps=grpo_cfg.logging_steps,
        save_steps=grpo_cfg.save_steps,
        save_total_limit=grpo_cfg.save_total_limit,
        max_grad_norm=grpo_cfg.max_grad_norm,
        weight_decay=grpo_cfg.weight_decay,
        optim=grpo_cfg.optim,
        report_to=grpo_cfg.report_to,
        seed=config.project.seed,
        # GRPO-specific
        num_generations=grpo_cfg.group_size,
        max_completion_length=grpo_cfg.max_completion_length,
        temperature=grpo_cfg.temperature,
    )

    # GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_training_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting GRPO training...")
    print("  (Each step renders Manim code - this is slower than SFT/DPO)")
    trainer.train()

    # Save
    print(f"\nSaving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    finish_wandb()
    print(f"\n=== GRPO Training Complete: {model_cfg.name} ===")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen2.5-coder-7b", "deepseek-coder-v2-lite", "codellama-7b"],
    )
    parser.add_argument("--dpo-checkpoint", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--kl-coeff", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.group_size:
        overrides["grpo.group_size"] = args.group_size
    if args.kl_coeff:
        overrides["grpo.kl_coeff"] = args.kl_coeff
    if args.lr:
        overrides["grpo.learning_rate"] = args.lr

    train(
        args.model,
        dpo_checkpoint=args.dpo_checkpoint,
        overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
