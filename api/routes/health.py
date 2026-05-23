"""Health check endpoint for the Generative Manim API."""

import os
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)

_PROVIDERS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "featherless": "FEATHERLESS_API_KEY",
    "litellm": "LITELLM_API_KEY",
}


def _check_providers() -> dict:
    return {
        name: {"configured": bool(os.getenv(env_var)), "env_var": env_var}
        for name, env_var in _PROVIDERS.items()
    }


@health_bp.route("/health", methods=["GET"])
@health_bp.route("/v1/health", methods=["GET"])
def health():
    providers = _check_providers()
    configured_count = sum(1 for p in providers.values() if p["configured"])
    status = "healthy" if configured_count > 0 else "degraded"

    return jsonify({
        "status": status,
        "providers": providers,
        "configured_providers": configured_count,
        "total_providers": len(providers),
    }), 200
