"""LocalModelEngine for direct integration with the Generative Manim API."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from ..utils.system_prompt import SYSTEM_PROMPT


class LocalModelEngine:
    """
    Drop-in replacement for OpenAI/Anthropic API calls in the GM API.

    Usage:
        engine = LocalModelEngine("./outputs/grpo/qwen2.5-coder-7b")
        code = engine.generate("Create a circle animation")

    Integration with Flask API:
        # In video_generation.py, add:
        if model.startswith("local-"):
            engine = LocalModelEngine(LOCAL_MODEL_PATH)
            code = engine.generate(prompt)
    """

    def __init__(
        self,
        checkpoint_path: str,
        base_model_id: str | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        self.checkpoint_path = checkpoint_path

        # Try to read base model from adapter config
        if base_model_id is None:
            import json
            from pathlib import Path
            adapter_config = Path(checkpoint_path) / "adapter_config.json"
            if adapter_config.exists():
                with open(adapter_config) as f:
                    config = json.load(f)
                base_model_id = config.get("base_model_name_or_path")

        if base_model_id is None:
            raise ValueError("base_model_id required (not found in adapter_config.json)")

        self.base_model_id = base_model_id

        # Load
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        """Generate Manim code from a text prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        return response
