import os

import anthropic
from flask import Blueprint, jsonify, request
from openai import OpenAI

from api.errors import bad_request, gateway_timeout, internal_error, not_found, rate_limited, unauthorized
from api.llm_providers import generate_gemini_content
from api.prompts.system import MANIM_CODE_GENERATION_PROMPT
from api.validation import get_json_body

code_generation_bp = Blueprint('code_generation', __name__)


ENGINE_DEFAULTS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5",
    "featherless": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "gemini": "gemini-2.5-flash",
    "litellm": "openai/gpt-4o",
}

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"


def get_openai_compatible_client(engine: str) -> OpenAI:
    if engine == "featherless":
        api_key = os.getenv("FEATHERLESS_API_KEY")
        if not api_key:
            raise ValueError("FEATHERLESS_API_KEY is required when engine='featherless'")
        return OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=api_key)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when engine='openai'")
    return OpenAI(api_key=api_key)


@code_generation_bp.route('/v1/code/generation', methods=['POST'])
def generate_code():
    body, err = get_json_body()
    if err:
        return err

    prompt_content = body.get("prompt", "")
    engine = body.get("engine", "openai")

    if not isinstance(engine, str) or engine not in ENGINE_DEFAULTS:
        return bad_request(
            f"Invalid engine. Must be one of: {', '.join(ENGINE_DEFAULTS.keys())}",
            code="invalid_engine",
        )

    model = body.get("model", ENGINE_DEFAULTS[engine])
    if model is not None and not isinstance(model, str):
        return bad_request("'model' must be a string", code="invalid_model")

    general_system_prompt = MANIM_CODE_GENERATION_PROMPT

    if engine == "litellm":
        import litellm
        from litellm.exceptions import AuthenticationError, NotFoundError, RateLimitError, Timeout

        messages = [
            {"role": "system", "content": general_system_prompt},
            {"role": "user", "content": prompt_content},
        ]
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "drop_params": True,
            }
            api_key = os.getenv("LITELLM_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key

            response = litellm.completion(**kwargs)
            code = response.choices[0].message.content
            return jsonify({"code": code})

        except AuthenticationError:
            return unauthorized("Authentication failed: check LITELLM_API_KEY", code="auth_failed")
        except NotFoundError:
            return not_found(
                "Model not found: use provider/model format (e.g. openai/gpt-4o)",
                code="model_not_found",
            )
        except RateLimitError:
            return rate_limited("LiteLLM rate limit exceeded", code="rate_limited")
        except Timeout:
            return gateway_timeout("LiteLLM request timed out", code="timeout")
        except Exception:
            return internal_error(code="litellm_error")

    elif engine == "anthropic" or model.startswith("claude-"):
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        messages = [{"role": "user", "content": prompt_content}]
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.2,
                system=general_system_prompt,
                messages=messages,
            )

            code = "".join(block.text for block in response.content)
            return jsonify({"code": code})

        except Exception:
            return internal_error(code="anthropic_error")

    elif engine == "gemini" or model.startswith("gemini-"):
        try:
            code = generate_gemini_content(model, general_system_prompt, prompt_content)
            return jsonify({"code": code})
        except Exception:
            return internal_error(code="gemini_error")

    else:
        messages = [
            {"role": "system", "content": general_system_prompt},
            {"role": "user", "content": prompt_content},
        ]

        try:
            client = get_openai_compatible_client(engine)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )

            code = response.choices[0].message.content
            return jsonify({"code": code})

        except Exception:
            return internal_error(code="openai_error")
