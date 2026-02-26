"""QLoRA DPO training with TRL DPOTrainer."""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

from ..utils.config_loader import load_config
from ..utils.logging_utils import setup_wandb, finish_wandb


def train(
    model_name: str,
    sft_checkpoint: str,
    overrides: dict | None = None,
):
    """Run DPO training on top of SFT checkpoint."""
    config = load_config(model_name=model_name, stage="dpo", overrides=overrides)
    dpo_cfg = config.dpo
    model_cfg = config.model

    print(f"=== DPO Training: {model_cfg.name} ===")
    print(f"  Base: {model_cfg.hf_id}")
    print(f"  SFT checkpoint: {sft_checkpoint}")
    print(f"  Beta: {dpo_cfg.beta}")

    # Setup W&B
    run_name = f"dpo-{model_cfg.name}"
    setup_wandb(config, run_name=run_name, tags=["dpo", model_cfg.name])

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

    # Load base model + SFT adapter as both model and reference
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # Load SFT adapter
    model = PeftModel.from_pretrained(base_model, sft_checkpoint, is_trainable=True)

    # LoRA config for DPO (new adapter on top)
    lora_cfg = config.lora
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    # Load DPO dataset
    dpo_path = config.paths.dpo_train

    def load_dpo_dataset(path):
        dataset = load_dataset("json", data_files=path, split="train")
        # DPOTrainer expects: prompt, chosen, rejected
        return dataset

    train_dataset = load_dpo_dataset(dpo_path)
    print(f"  DPO pairs: {len(train_dataset)}")

    # Output directory
    output_dir = os.path.join(dpo_cfg.output_dir, model_cfg.name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg.num_train_epochs,
        learning_rate=dpo_cfg.learning_rate,
        lr_scheduler_type=dpo_cfg.lr_scheduler_type,
        warmup_ratio=dpo_cfg.warmup_ratio,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        fp16=dpo_cfg.fp16,
        bf16=dpo_cfg.bf16,
        logging_steps=dpo_cfg.logging_steps,
        save_steps=dpo_cfg.save_steps,
        save_total_limit=dpo_cfg.save_total_limit,
        max_grad_norm=dpo_cfg.max_grad_norm,
        weight_decay=dpo_cfg.weight_decay,
        optim=dpo_cfg.optim,
        report_to=dpo_cfg.report_to,
        seed=config.project.seed,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Uses implicit reference (frozen copy)
        args=training_args,
        beta=dpo_cfg.beta,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_prompt_length=dpo_cfg.max_prompt_length,
        max_length=dpo_cfg.max_length,
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    # Save
    print(f"\nSaving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    finish_wandb()
    print(f"\n=== DPO Training Complete: {model_cfg.name} ===")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen2.5-coder-7b", "deepseek-coder-v2-lite", "codellama-7b"],
    )
    parser.add_argument("--sft-checkpoint", type=str, required=True)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.beta:
        overrides["dpo.beta"] = args.beta
    if args.lr:
        overrides["dpo.learning_rate"] = args.lr

    train(
        args.model,
        sft_checkpoint=args.sft_checkpoint,
        overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
