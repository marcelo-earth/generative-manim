from flask import Blueprint, jsonify, request
import anthropic
import os
from openai import OpenAI
from api.llm_providers import generate_gemini_content
from api.validation import get_json_body

code_generation_bp = Blueprint('code_generation', __name__)


ENGINE_DEFAULTS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
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
        return jsonify({
            "error": f"Invalid engine. Must be one of: {', '.join(ENGINE_DEFAULTS.keys())}"
        }), 400

    model = body.get("model", ENGINE_DEFAULTS[engine])
    if model is not None and not isinstance(model, str):
        return jsonify({"error": "'model' must be a string"}), 400

    general_system_prompt = """
You are an assistant that knows about Manim. Manim is a mathematical animation engine that is used to create videos programmatically.

The following is an example of the code:
\`\`\`
from manim import *
from math import *

class GenScene(Scene):
def construct(self):
    c = Circle(color=BLUE)
    self.play(Create(c))

\`\`\`

# Rules
1. Always use GenScene as the class name, otherwise, the code will not work.
2. Always use self.play() to play the animation, otherwise, the code will not work.
3. Do not use text to explain the code, only the code.
4. Do not explain the code, only the code.
    """

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

        except AuthenticationError as e:
            return jsonify({"error": f"LiteLLM auth failed (check API key): {e}"}), 401
        except NotFoundError as e:
            return jsonify({"error": f"LiteLLM model not found (use provider/model format, e.g. openai/gpt-4o): {e}"}), 404
        except RateLimitError as e:
            return jsonify({"error": f"LiteLLM rate limit exceeded: {e}"}), 429
        except Timeout as e:
            return jsonify({"error": f"LiteLLM request timed out: {e}"}), 504
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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

            # Extract the text content from the response
            code = "".join(block.text for block in response.content)

            return jsonify({"code": code})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif engine == "gemini" or model.startswith("gemini-"):
        try:
            code = generate_gemini_content(model, general_system_prompt, prompt_content)
            return jsonify({"code": code})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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

        except Exception as e:
            return jsonify({"error": str(e)}), 500
