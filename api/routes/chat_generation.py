import base64
import io
import json
import os
import random
import re
import shutil
import string
import subprocess
import time

import anthropic
import openai
from flask import Blueprint, Response, jsonify, request, stream_with_context
from openai import APIError
from PIL import Image

from api.llm_providers import generate_gemini_content_stream
from api.prompts.manimDocs import manimDocs
from api.validation import get_json_body

chat_generation_bp = Blueprint("chat_generation", __name__)

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

_ENGINE_DEFAULTS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5",
    "deepseek": "r1",
    "featherless": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "litellm": "openai/gpt-4o",
    "gemini": "gemini-2.5-flash",
}

_VALID_MODELS = {
    "openai": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-4o", "o1-mini"],
    "anthropic": [
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-35-sonnet",
    ],
    "deepseek": ["r1"],
    "featherless": None,
    "litellm": None,
    "gemini": None,
}


animo_functions = {
    "openai": [
        {
            "name": "get_preview",
            "description": "Get a preview of the video animation before giving it. Use this function always, before giving the final code to the user. And use it to generate frames of the video, so you can see it and improve it over time. Also, before using this function, tell the user you will be generating a preview based on the code they see. Always use spaces to maintain the indentation. Indentation is important, otherwise the code will not work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to get the preview of. Take account the spaces to maintain the indentation.",
                    },
                    "class_name": {
                        "type": "string",
                        "description": "The name of the class to get the preview of. The name of the class should be the same as the name of the class in the code.",
                    }
                },
                "required": ["code", "class_name"],
            },
            "output": {"type": "string", "description": "Images URLs of the animation that will be inserted in the conversation"},
        }
    ],
    "anthropic": [
        {
            "name": "get_preview",
            "description": "Get a preview of the video animation before giving it. Use this function always, before giving the final code to the user. And use it to generate frames of the video, so you can see it and improve it over time. Also, before using this function, tell the user you will be generating a preview based on the code they see. Always use spaces to maintain the indentation. Indentation is important, otherwise the code will not work.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to get the preview of. Take account the spaces to maintain the indentation."
                    },
                    "class_name": {
                        "type": "string",
                        "description": "The name of the class to get the preview of. The name of the class should be the same as the name of the class in the code."
                    }
                },
                "required": ["code", "class_name"]
            }
        }
    ]
}

def count_images_in_conversation(messages):
    total_images = 0
    image_message_indices = []
    for i, message in enumerate(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            image_count = sum(
                1 for content in message["content"]
                if isinstance(content, dict) and content.get("type") == "image_url"
            )
            if image_count > 0:
                total_images += image_count
                image_message_indices.append(i)
    return total_images, image_message_indices

def manage_conversation_images(messages, new_images_count, engine):
    if engine != "openai":
        return new_images_count

    MAX_IMAGES = 50
    current_total, image_indices = count_images_in_conversation(messages)

    while current_total > 0 and current_total + new_images_count > MAX_IMAGES and image_indices:
        oldest_image_idx = image_indices[0]
        removed_message = messages.pop(oldest_image_idx)
        removed_images = sum(1 for content in removed_message["content"]
                             if isinstance(content, dict) and content.get("type") == "image_url")
        current_total -= removed_images
        image_indices = [idx - 1 for idx in image_indices[1:]]

    return min(MAX_IMAGES - current_total, new_images_count)


def _generate_manim_preview(code: str, class_name: str) -> str:
    """Run Manim in PNG mode and return a JSON string with base64-encoded frames."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.dirname(current_dir)

    temp_dir = os.path.join(api_dir, "temp_manim")
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, f"{class_name}.py")
    preview_code = f"from manim import *\nfrom math import *\n\n{code}\n"

    with open(file_path, "w") as f:
        f.write(preview_code)

    command = (
        f"manim {file_path} {class_name} "
        f"--format=png --media_dir {temp_dir} --custom_folders -pql --disable_caching"
    )
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        previews_dir = os.path.join(api_dir, "public", "previews")
        os.makedirs(previews_dir, exist_ok=True)
        random_string = "".join(random.choices(string.ascii_letters + string.digits, k=12))
        destination_dir = os.path.join(previews_dir, random_string, class_name)

        png_files = [f for f in os.listdir(temp_dir) if f.endswith(".png")]
        if png_files:
            os.makedirs(destination_dir, exist_ok=True)
            image_list = []
            for png_file in png_files:
                shutil.move(os.path.join(temp_dir, png_file), os.path.join(destination_dir, png_file))
                match = re.search(r"(\d+)\.png$", png_file)
                if match:
                    index = int(match.group(1))
                    if index % 4 == 0:
                        image_path = os.path.join(destination_dir, png_file)
                        with Image.open(image_path) as img:
                            new_w, new_h = img.size[0] // 4, img.size[1] // 4
                            resized = img.resize((new_w, new_h), Image.LANCZOS)
                            buf = io.BytesIO()
                            resized.save(buf, format="PNG")
                            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        image_list.append({"path": image_path, "index": index, "base64": b64})
            image_list.sort(key=lambda x: x["index"])
            return json.dumps({
                "message": "Animation preview generated. Now you will see the image frames in the next automatic message...",
                "images": image_list,
            })
        else:
            return json.dumps({"error": f"No preview files generated at: {temp_dir}", "images": []})

    except subprocess.CalledProcessError as e:
        error_output = e.stdout + e.stderr
        return json.dumps({
            "error": f"ERROR. Error generating preview, please think on what could be the problem, and use `get_preview` to run the code again: {e}\nCommand output:\n{error_output}",
            "images": [],
        })
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}", "images": []})


def _streaming_response(generate_fn, is_for_platform: bool):
    """Wrap a generator in a Flask streaming Response with correct SSE headers."""
    response = Response(
        stream_with_context(generate_fn()),
        content_type="text/plain; charset=utf-8",
    )
    if is_for_platform:
        response.headers["Transfer-Encoding"] = "chunked"
        response.headers["x-vercel-ai-data-stream"] = "v1"
    return response


@chat_generation_bp.route("/v1/chat/generation", methods=["POST"])
def generate_code_chat():
    data, err = get_json_body()
    if err:
        return err

    messages = data.get("messages", [])
    prompt = data.get("prompt")
    engine = data.get("engine", "openai")
    model = data.get("model", None)
    is_for_platform = data.get("isForPlatform", False)

    if engine not in _ENGINE_DEFAULTS:
        return jsonify({"error": f"Invalid engine. Must be one of: {', '.join(_ENGINE_DEFAULTS.keys())}"}), 400

    if not model:
        model = _ENGINE_DEFAULTS[engine]

    if _VALID_MODELS[engine] is not None and model not in _VALID_MODELS[engine]:
        return jsonify({
            "error": f"Invalid model '{model}' for engine '{engine}'. Valid models are: {', '.join(_VALID_MODELS[engine])}"
        }), 400

    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]

    general_system_prompt = r"""You are an assistant that creates animations with Manim. Manim is a mathematical animation engine that is used to create videos programmatically. You are running on Animo (www.animo.video), a tool to create videos with Manim.

# What the user can do?

The user can create a new project, add scenes, and generate the video. You can help the user to generate the video by creating the code for the scenes. The user can add custom rules for you, can select a different aspect ratio, and can change the model (the models are: OpenAI GPT-5.6, and Anthropic Claude Sonnet 5).

# Project

A project can be composed of multiple scenes. This current project (where the user is working on right now) is called '{project_title}', and the following scenes are part of this project. The purpose of showing the list of scenes is to keep the context of the whole video project.

## List of scenes:
{scenes_prompt}

# Behavior Context

The user will ask you to generatte an animation, you should iterate while using the `get_preview` function. This function will generate a preview of the animation, and will be inserted in the conversation so you can see the frames of it, and enhance it across the time. You can make this iteration up to 4 times without user confirmation. Just use the `get_preview` until you are sure that the animation is ready.

FAQ
**Should the assistant generate the code first and then use the `get_preview` function?**
No, unless the user asks for it. Always use the `get_preview` function to generate the code. The user will see the code anyway, so there is no need to duplicate the work. Use the `get_preview` as your way to quickly draft the code and then iterate on it.

**Can the user see the code generated?**
Yes, even if you use the `get_preview` function, the code will be generated and visible for the user.

**Can the assistant propose a more efficient way to generate the animation?**
Yes, the assistant can propose a more efficient way to generate the animation. For example, the assistant can propose a different aspect ratio, a different model, or a different scene. If the change is too big, you should ask the user for confirmation. Act with initiative.

**Should the assistant pause to change the code?**
Yes, always stay in the loop of generating the preview and improving it from what you see.
Incorrect: Please hold on while I make these adjustments.
Correct: Now I will do the adjustments *and does the adjustments*.

**Should the assistant tell the user about the get_preview function?**
Yes, here are some examples:

1. Let me generate a preview of the animation to see how it looks like for you. I'll see it.
2. I have an idea on how to improve the animation, let me visualize it for a second.
3. OK. Now I know how to improve the animation. Please give me a moment to preview it.

# Code Context

The following is an example of the code:
\`\`\`
from manim import *
from math import *

class GenScene(Scene):
  def construct(self):
      # Create a circle of color BLUE
      c = Circle(color=BLUE)
      # Play the animation of creating the circle
      self.play(Create(c))

\`\`\`

The following is an example of the code with a voiceover:

\`\`\`
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim.scene.scene_file_writer import SceneFileWriter

# Disable the creation of subcaption files only when using Text-to-Speech
SceneFileWriter.write_subcaption_file = lambda *args, **kwargs: None

class GenScene(VoiceoverScene):
    def construct(self):
        # Set the voiceover service to Google Text-to-Speech
        self.set_speech_service(GTTSService())
        
        # Create a circle of color BLUE
        c = Circle(color=BLUE)
        c.move_to(ORIGIN)

        # Voiceover with animation
        with self.voiceover(
            text="This circle is drawn as I speak.",
            subcaption="What a cute circle! :)"
        ) as tracker:
            self.play(Create(c), run_time=tracker.duration)
\`\`\`

Let the user know that they can hear the audio of the voiceover by clicking on the "Animate" button.

Remember the part of `from manim.scene.scene_file_writer import SceneFileWriter` `SceneFileWriter.write_subcaption_file = lambda *args, **kwargs: None` because it's important to not create a subcaption file.

## Rules of programming

1. Always use comments to explain the next line of the code:

\`\`\`
# Create a sphere of color BLUE for the Earth
earth = Sphere(radius=1, checkerboard_colors=[BLUE_D, BLUE_E])
\`\`\`

This is needed to understand what you meant to do.

2. You can use TODO comments to mark places that you think can be improved, and you can come back later to them.

\`\`\`
# TODO: Add more colors to the cube later
\`\`\`

This is needed to understand what you could have done better.

3. Everytime there is a movement on the camera or on the objects, you should add a comment resalting the desired movement

\`\`\`
# With this movement we should see the difference between the both buildings
self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
\`\`\`

This is needed to understand what you meant to reflect on the camera.

4. Unless described on this system prompt, always assume the user is not providing any image to you. That means, you should not use, for example:

\`\`\`
# Create a 3D grid of spheres
new_texture = "assets/random_texture.jpg"
\`\`\`

If `random_texture.jpg` is not provided, you should not use it. Otherwise the video will not work on the platform.

5. At the very end of all the thinking process, you should provide the final code to the user in a codeblock (without the ```python at the beginning and the end). And tell the user to click on "Animate" to see the video. Like this:

- ...Finally, you can hit "Animate" to render the video at the right side.
- ...Now to see the animation, click on "Animate".
- ...Cool, that's all! Now click on "Animate" to see the video.

That message should appear after the code, as the last message of the conversation.

6. If the user request a complex animation or an animation of a complex component, scene or object. Before generating the code, you need to think about it, these are some good examples to follow:

- User: I want to animate a car.
- Assistant (You): Okay, I need to generate Manim code for a car. Let me start by thinking about how to represent a car using Manim's shapes. A car typically has a body, wheels, windows, and maybe some details like headlights. Let's break it down into parts. First, the car body. Maybe a rectangle for the main part. Then a rounded rectangle or a polygon for the top part to make it look more like a car. I can use a Rectangle or RoundedRectangle class for the body. Maybe start with a rectangle and then add a trapezoid or something on top for the roof. Next, the wheels. Circles would work for wheels. I should place them at the bottom of the car body. Two circles on each side, maybe. To make them look like tires, perhaps add a smaller circle inside for the hubcap. Windows. The car probably has a front and rear window. Maybe two trapezoids or rectangles on top of the car body. Or combine them into a single shape with a line dividing them. Headlights and other details. Small circles or rectangles at the front and back.

- User: Create a Solar System.
- Assistant (You): Let's think about how to represent a Solar System using Manim's shapes. A Solar System typically has a Sun, planets, and maybe some details like moons. Let's break it down into parts. First, the Sun. Maybe a circle for the Sun. Then, the planets. Each planet is a circle, and they orbit around the Sun. Maybe start with the inner planets and then add the outer planets. Next, the moons. Each moon is a smaller circle that orbits around its planet. Also I should add a line to represent the orbit of each planet. The color of the planets should be different, and the color of the Sun should be yellow.

7. Always use `GenScene` as the class name, unless the user asks for a different name. Use `GenScene` as the class name by default.

## Rules of behaviour

1. If the user just says Hello, or something like that, you should not generate any code. Just tell the user you are ready to help them. Show what you can do, suggest what you can do.

2. Always be very kind, you can use words (in English, or translated to other languages) like:
- You're right!...
- That's awesome...
- Yes, I think I can do that!...
- Ah! That's true!, sorry...

3. If the user have a question you can't answer about Animo (the platform animo.video), you can tell them to send a message to team@animo.video with a clear description of the question or problem. Tell them we're back to them in less than 24 hours.

# Manim Library
{manimDocs}
"""

    messages.insert(0, {"role": "system", "content": general_system_prompt})

    if engine == "litellm":
        import litellm

        litellm_system_prompt = general_system_prompt
        messages[0] = {"role": "system", "content": litellm_system_prompt}

        def generate():
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": data.get("temperature", 0.2),
                    "max_tokens": data.get("maxTokens", 2048),
                    "stream": True,
                    "drop_params": True,
                }
                api_key = os.getenv("LITELLM_API_KEY")
                if api_key:
                    kwargs["api_key"] = api_key

                stream = litellm.completion(**kwargs)
                for chunk in stream:
                    if not chunk.choices or not chunk.choices[0].delta:
                        continue
                    content = chunk.choices[0].delta.content
                    if not content:
                        continue
                    if is_for_platform:
                        text_obj = json.dumps({"type": "text", "text": content})
                        yield f"{text_obj}\n"
                    else:
                        yield content
            except Exception as e:
                error_message = str(e)
                if is_for_platform:
                    yield f"{json.dumps({'type': 'error', 'text': error_message})}\n"
                else:
                    yield f"Error: {error_message}"

        return _streaming_response(generate, is_for_platform)

    elif engine == "featherless":
        api_key = os.environ.get("FEATHERLESS_API_KEY")
        if not api_key:
            return jsonify({"error": "FEATHERLESS_API_KEY is required when engine='featherless'"}), 500

        client = openai.OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=api_key)
        featherless_system_prompt = """You are an assistant that generates complete Manim animation code.

Rules:
1. Always use GenScene as the class name.
2. Always use self.play() to play animations.
3. Always start with `from manim import *`.
4. Output only Python code unless the user asks a conceptual question.
5. The code must be complete and runnable."""
        messages[0] = {"role": "system", "content": featherless_system_prompt}

        def generate():
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=data.get("temperature", 0.2),
                    max_tokens=data.get("maxTokens", 2048),
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if not content:
                        continue
                    if is_for_platform:
                        text_obj = json.dumps({"type": "text", "text": content})
                        yield f"{text_obj}\n"
                    else:
                        yield content
            except Exception as e:
                safe_message = f"{type(e).__name__}: generation failed"
                if is_for_platform:
                    yield f"{json.dumps({'type': 'error', 'text': safe_message})}\n"
                else:
                    yield f"Error: {safe_message}"

        return _streaming_response(generate, is_for_platform)

    if engine == "gemini":
        gemini_system_prompt = general_system_prompt

        def generate():
            try:
                for chunk in generate_gemini_content_stream(model, gemini_system_prompt, messages):
                    if is_for_platform:
                        text_obj = json.dumps({"type": "text", "text": chunk})
                        yield f"{text_obj}\n"
                    else:
                        yield chunk
            except Exception as e:
                safe_message = f"{type(e).__name__}: generation failed"
                if is_for_platform:
                    yield f"{json.dumps({'type': 'error', 'text': safe_message})}\n"
                else:
                    yield f"Error: {safe_message}"

        return _streaming_response(generate, is_for_platform)

    if engine == "anthropic":
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        get_preview = _generate_manim_preview

        def convert_message_for_anthropic(message):
            if isinstance(message["content"], list):
                content = []
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        content.append(
                            {"type": "image", "image": part["image_url"]["url"]}
                        )
                    else:
                        content.append(part)
                message["content"] = content
            return message

        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        anthropic_messages = [
            convert_message_for_anthropic(msg)
            for msg in messages
            if msg["role"] != "system"
        ]

        def generate():
            try:
                messages = anthropic_messages
                while True:
                    stream = client.messages.create(
                        model=model,
                        messages=messages,
                        system=system_message,
                        max_tokens=1000,
                        stream=True,
                        tools=animo_functions["anthropic"]
                    )
                    
                    current_message = {"role": "assistant", "content": []}
                    current_text = ""
                    should_continue = False
                    tool_use_id = None
                    complete_json = ""
                    
                    for chunk in stream:
                        if chunk.type == "content_block_start":
                            if hasattr(chunk.content_block, 'type'):
                                if chunk.content_block.type == 'tool_use':
                                    tool_use_id = chunk.content_block.id
                                    if current_text:
                                        current_message["content"].append({
                                            "type": "text",
                                            "text": current_text
                                        })
                                        current_text = ""
                                    tool_input = {}
                                    if complete_json:
                                        try:
                                            tool_input = json.loads(complete_json)
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    current_message["content"].append({
                                        "type": "tool_use",
                                        "id": tool_use_id,
                                        "name": "get_preview",
                                        "input": tool_input
                                    })
                        
                        elif chunk.type == "content_block_delta":
                            if hasattr(chunk.delta, 'text'):
                                content = chunk.delta.text
                                if content:
                                    current_text += content
                                    if is_for_platform:
                                        for char in content:
                                            escaped_char = repr(char)[1:-1]
                                            yield f'0:"{escaped_char}"\n'
                                    else:
                                        yield content
                                
                            elif hasattr(chunk.delta, 'partial_json'):
                                complete_json += chunk.delta.partial_json

                        elif chunk.type == "content_block_stop":
                            if complete_json:
                                try:
                                    tool_call = json.loads(complete_json)
                                    preview_result = get_preview(
                                        code=tool_call.get('code', ''),
                                        class_name=tool_call.get('class_name', '')
                                    )
                                    try:
                                        preview_data = json.loads(preview_result)
                                        middle_frame = preview_data['images'][len(preview_data['images'])//2]
                                        base64_data = middle_frame['base64']
                                        content_blocks = [
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": "image/png",
                                                    "data": base64_data,
                                                }
                                            },
                                            {
                                                "type": "text",
                                                "text": "\nPreview frame from the animation.\n"
                                            }
                                        ]
                                        tool_response = {
                                            "role": "user",
                                            "content": [{
                                                "type": "tool_result",
                                                "tool_use_id": tool_use_id,
                                                "content": content_blocks
                                            }]
                                        }
                                        messages.append(current_message)
                                        messages.append(tool_response)
                                        should_continue = True
                                        
                                        preview_text = "Generated preview of the animation:\n"
                                        if is_for_platform:
                                            for char in preview_text:
                                                escaped_char = repr(char)[1:-1]
                                                yield f'0:"{escaped_char}"\n'
                                            yield '0:"[IMAGE: Preview frame]"\n'
                                        else:
                                            yield "\n[Preview frame]\n"
                                        
                                    except json.JSONDecodeError:
                                        continue
                                except Exception:
                                    continue

                        elif chunk.type == "message_stop":
                            if current_text:
                                if not current_message["content"]:
                                    current_message["content"] = []
                                current_message["content"].append({
                                    "type": "text",
                                    "text": current_text
                                })
                                messages.append(current_message)
                            
                            if not should_continue:
                                return
                            break
                    
                    if not should_continue:
                        break

            except Exception as e:
                error_message = f'0:"{str(e)}"\n' if is_for_platform else f"Error: {str(e)}"
                yield error_message

        return _streaming_response(generate, is_for_platform)

    else:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        get_preview = _generate_manim_preview

        def generate():
            max_retries = 3
            retry_delay = 4  # seconds

            while True:
                for attempt in range(max_retries):
                    try:
                        stream = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=True,
                            functions=animo_functions["openai"],
                            function_call="auto",
                        )
                        function_call_data = ""
                        function_name = ""
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                if is_for_platform:
                                    text_obj = json.dumps({"type": "text", "text": content})
                                    yield f'{text_obj}\n'
                                else:
                                    yield content
                            elif chunk.choices[0].delta.function_call:
                                if chunk.choices[0].delta.function_call.name:
                                    function_name = chunk.choices[0].delta.function_call.name
                                    if is_for_platform:
                                        initial_call_obj = json.dumps({
                                            "type": "function_call",
                                            "content": "",
                                            "function_call": {"name": function_name}
                                        })
                                        yield f'{initial_call_obj}\n'
                                if chunk.choices[0].delta.function_call.arguments:
                                    chunk_data = chunk.choices[0].delta.function_call.arguments
                                    function_call_data += chunk_data
                                    if is_for_platform:
                                        partial_call_obj = json.dumps({
                                            "type": "function_call",
                                            "content": "",
                                            "function_call": {"args": chunk_data}
                                        })
                                        yield f'{partial_call_obj}\n'
                        
                        break
                    
                    except APIError:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            yield json.dumps({"error": "Max retries reached due to API errors"})
                            return

                if function_call_data:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": function_call_data
                        }
                    })

                    if function_name == "get_preview":
                        args = json.loads(function_call_data)
                        result = get_preview(args['code'], args['class_name'])
                        result_json = json.loads(result)
                        function_response = {
                            "content": result_json.get("message", result_json.get("error")),
                            "name": "get_preview",
                            "role": "function"
                        }
                        messages.append(function_response)

                        if is_for_platform:
                            function_result_obj = json.dumps({
                                "type": "function_result",
                                "content": function_response,
                                "function_call": {"name": function_name}
                            })
                            yield f'{function_result_obj}\n'

                        if result_json.get("images"):
                            image_message = {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """ASSISTANT_MESSAGE_PREVIEW_GENERATED: This message is not generated by the user, but automatically by you, the assistant when firing the `get_preview` function, this message might not be visible to the user.
                                        
                                        The following images are selected frames of the animation generated. Please check these frames and follow the rules: Text should not be overlapping, the space should be used efficiently, use different colors to represent different objects, plus other improvements you can think of.
                                        
                                        You can decide now if you want to iterate on the animation (if it's too complex), or just stop here and provide the final code to the user now."""
                                    }
                                ]
                            }

                            available_slots = manage_conversation_images(messages, len(result_json["images"]), engine)
                            total_frames = len(result_json["images"])
                            frame_interval = max(1, total_frames // available_slots)
                            selected_frames = result_json["images"][::frame_interval][:available_slots]

                            for image in selected_frames:
                                image_message["content"].append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image['base64']}"
                                    }
                                })
                            messages.append(image_message)

                            if not is_for_platform:
                                yield json.dumps(image_message)

                        continue
                    else:
                        break
                else:
                    break

            final_message = "\n"
            if is_for_platform:
                text_obj = json.dumps({"type": "text", "text": final_message})
                yield f'{text_obj}\n'
            else:
                yield final_message

        return _streaming_response(generate, is_for_platform)
