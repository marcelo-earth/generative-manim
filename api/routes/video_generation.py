import os
import shutil
import subprocess
import uuid

import anthropic
from flask import Blueprint, jsonify, request
from openai import OpenAI

from api.errors import gateway_timeout, internal_error
from api.llm_providers import generate_gemini_content
from api.prompts.system import MANIM_CODE_GENERATION_PROMPT
from api.validation import get_json_body, require_string, validate_aspect_ratio
from api.video_utils import get_frame_config

video_generation_bp = Blueprint('video_generation', __name__)


def generate_manim_code(prompt, engine="openai", model="gpt-5.6-terra"):
    if model.startswith("claude-"):
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        messages = [{"role": "user", "content": prompt}]
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.2,
                system=MANIM_CODE_GENERATION_PROMPT,
                messages=messages,
            )
            code = "".join(block.text for block in response.content)
            return code
        except Exception as e:
            raise Exception(f"Error generating code with {model}: {str(e)}")
    elif engine == "gemini" or model.startswith("gemini-"):
        try:
            return generate_gemini_content(model, MANIM_CODE_GENERATION_PROMPT, prompt)
        except Exception as e:
            raise Exception(f"Error generating code with {model}: {str(e)}")
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": MANIM_CODE_GENERATION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            code = response.choices[0].message.content
            return code
        except Exception as e:
            raise Exception(f"Error generating code with {model}: {str(e)}")


@video_generation_bp.route("/v1/video/generation", methods=["POST"])
def generate_video():
    body, err = get_json_body()
    if err:
        return err

    try:
        prompt, err = require_string(body, "prompt")
        if err:
            return err

        aspect_ratio, err = validate_aspect_ratio(body)
        if err:
            return err

        engine = body.get("engine", "openai")
        model = body.get("model", "gpt-5.6-terra")
        user_id = body.get("user_id") or str(uuid.uuid4())
        project_name = body.get("project_name", "untitled")
        iteration = body.get("iteration", 1)

        try:
            code = generate_manim_code(prompt, engine, model)
        except Exception:
            return internal_error("Code generation failed", code="code_generation_failed")

        try:
            frame_size, frame_width = get_frame_config(aspect_ratio)
            modified_code = f"""
from manim import *
from math import *
config.frame_size = {frame_size}
config.frame_width = {frame_width}

{code}
            """

            file_name = f"scene_{os.urandom(2).hex()}.py"
            api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            public_dir = os.path.join(api_dir, "public")
            os.makedirs(public_dir, exist_ok=True)
            file_path = os.path.join(public_dir, file_name)

            with open(file_path, "w") as f:
                f.write(modified_code)

            command = [
                "manim",
                file_path,
                "GenScene",
                "--format=mp4",
                "--media_dir",
                ".",
                "--custom_folders",
            ]

            result = subprocess.run(
                command,
                cwd=os.path.dirname(os.path.realpath(__file__)),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return internal_error("Manim rendering failed", code="render_failed")

            video_file_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "GenScene.mp4"
            )

            if not os.path.exists(video_file_path):
                return internal_error("Video file not found after rendering", code="video_not_found")

            video_storage_file_name = f"video-{user_id}-{project_name}-{iteration}"
            new_file_name = f"{video_storage_file_name}.mp4"
            new_file_path = os.path.join(public_dir, new_file_name)

            shutil.move(video_file_path, new_file_path)

            base_url = request.host_url.rstrip('/') if request.host_url else os.getenv("BASE_URL", "http://127.0.0.1:8080")
            video_url = f"{base_url}/public/{new_file_name}"

            return jsonify({
                "message": "Video generated successfully",
                "video_url": video_url,
                "code": code
            }), 200

        except subprocess.TimeoutExpired:
            return gateway_timeout("Video rendering timed out", code="render_timeout")
        except Exception:
            return internal_error("Video rendering failed", code="render_error")
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)
            except Exception:
                pass

    except Exception:
        return internal_error(code="unexpected_error")
