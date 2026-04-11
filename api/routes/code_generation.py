from flask import Blueprint, jsonify, request
import anthropic
import os
from openai import OpenAI
try:
    from google import genai
except ImportError:
    genai = None

code_generation_bp = Blueprint('code_generation', __name__)

@code_generation_bp.route('/v1/code/generation', methods=['POST'])
def generate_code():
    body = request.json
    prompt_content = body.get("prompt", "")
    model = body.get("model", "gpt-4o")

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

    if model.startswith("claude-"):
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

    elif model.startswith("gemini-"):
        if genai is None:
            return jsonify({"error": "google-genai pkg not installed"}), 500
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        try:
            # We map standard gemini aliases
            if model == "gemini-1.5-pro":
                use_model = "gemini-1.5-pro"
            elif model == "gemini-2.0-flash":
                use_model = "gemini-2.5-flash" # fallback or actual name when released
            else:
                use_model = model

            response = client.models.generate_content(
                model=use_model,
                contents=f"{general_system_prompt}\n\nUser request: {prompt_content}",
            )
            # Remove any markdown code block artifacts if present
            code = response.text
            if code.startswith("```"):
                code = "\n".join(code.split("\n")[1:])
            if code.endswith("```"):
                code = "\n".join(code.split("\n")[:-1])
            return jsonify({"code": code.strip()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": general_system_prompt},
            {"role": "user", "content": prompt_content},
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )

            code = response.choices[0].message.content

            return jsonify({"code": code})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
