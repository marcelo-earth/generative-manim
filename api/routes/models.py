"""List available models grouped by engine."""

import os
from flask import Blueprint, jsonify

models_bp = Blueprint("models", __name__)

_ENGINES = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "default": "gpt-4o",
        "models": [
            {"id": "gpt-4o", "description": "GPT-4o — OpenAI's latest multimodal model"},
            {"id": "o1-mini", "description": "o1-mini — compact reasoning model"},
        ],
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "default": "claude-sonnet-4-6",
        "models": [
            {"id": "claude-sonnet-4-6", "description": "Claude Sonnet 4.6 — balanced speed and quality"},
            {"id": "claude-opus-4-7", "description": "Claude Opus 4.7 — highest capability"},
            {"id": "claude-haiku-4-5-20251001", "description": "Claude Haiku 4.5 — fastest and most compact"},
            {"id": "claude-3-5-sonnet-20241022", "description": "Claude 3.5 Sonnet (legacy)"},
        ],
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "default": "gemini-2.5-flash",
        "models": [
            {"id": "gemini-2.5-flash", "description": "Gemini 2.5 Flash — fast and efficient"},
            {"id": "gemini-2.5-pro", "description": "Gemini 2.5 Pro — highest capability"},
        ],
    },
    "featherless": {
        "env_var": "FEATHERLESS_API_KEY",
        "default": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "models": [
            {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "description": "Qwen 2.5 Coder 7B"},
            {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "description": "Qwen 2.5 Coder 32B"},
            {"id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "description": "DeepSeek Coder V2 Lite"},
            {"id": "meta-llama/CodeLlama-7b-Instruct-hf", "description": "CodeLlama 7B"},
        ],
    },
    "litellm": {
        "env_var": "LITELLM_API_KEY",
        "default": "openai/gpt-4o",
        "models": [],
        "note": "Accepts any model string supported by LiteLLM (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-6).",
    },
}


@models_bp.route("/v1/models", methods=["GET"])
def list_models():
    result = []
    for engine, info in _ENGINES.items():
        configured = bool(os.getenv(info["env_var"]))
        entry = {
            "engine": engine,
            "configured": configured,
            "default": info["default"],
            "models": info["models"],
        }
        if "note" in info:
            entry["note"] = info["note"]
        result.append(entry)

    return jsonify({"engines": result}), 200
