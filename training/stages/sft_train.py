"""QLoRA SFT training with TRL SFTTrainer."""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from ..utils.config_loader import load_config
from ..utils.logging_utils import setup_wandb, finish_wandb


def train(model_name: str, overrides: dict | None = None):
    """Run SFT training for a given model."""
    config = load_config(model_name=model_name, stage="sft", overrides=overrides)
    sft_cfg = config.sft
    model_cfg = config.model

    print(f"=== SFT Training: {model_cfg.name} ===")
    print(f"  Model: {model_cfg.hf_id}")
    print(f"  Epochs: {sft_cfg.num_train_epochs}")
    print(f"  Learning rate: {sft_cfg.learning_rate}")

    # Setup W&B
    run_name = f"sft-{model_cfg.name}"
    setup_wandb(config, run_name=run_name, tags=["sft", model_cfg.name])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.qlora.load_in_4bit,
        bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_cfg = config.lora
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    # Load dataset
    train_path = config.paths.sft_train
    val_path = config.paths.sft_val

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train")

    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")

    # Format function for chat template
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    # Output directory
    output_dir = os.path.join(sft_cfg.output_dir, model_cfg.name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.num_train_epochs,
        learning_rate=sft_cfg.learning_rate,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        warmup_ratio=sft_cfg.warmup_ratio,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        fp16=sft_cfg.fp16,
        bf16=sft_cfg.bf16,
        logging_steps=sft_cfg.logging_steps,
        save_steps=sft_cfg.save_steps,
        eval_steps=sft_cfg.eval_steps,
        eval_strategy="steps",
        save_total_limit=sft_cfg.save_total_limit,
        max_grad_norm=sft_cfg.max_grad_norm,
        weight_decay=sft_cfg.weight_decay,
        optim=sft_cfg.optim,
        group_by_length=sft_cfg.group_by_length,
        report_to=sft_cfg.report_to,
        seed=config.project.seed,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        max_seq_length=model_cfg.max_seq_length,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting SFT training...")
    trainer.train()

    # Save
    print(f"\nSaving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    finish_wandb()
    print(f"\n=== SFT Training Complete: {model_cfg.name} ===")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen2.5-coder-7b", "deepseek-coder-v2-lite", "codellama-7b"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["sft.num_train_epochs"] = args.epochs
    if args.lr:
        overrides["sft.learning_rate"] = args.lr
    if args.batch_size:
        overrides["sft.per_device_train_batch_size"] = args.batch_size

    train(args.model, overrides=overrides if overrides else None)


if __name__ == "__main__":
    main()
