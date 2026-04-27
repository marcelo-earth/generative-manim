from flask import Blueprint, jsonify, request, Response, stream_with_context
import anthropic
import openai
import os
import json
import subprocess
import shutil
import string
import random
import re
import base64
from api.prompts.manimDocs import manimDocs
from azure.storage.blob import BlobServiceClient
from PIL import Image
import io
import time
from openai import APIError
import uuid

chat_generation_bp = Blueprint("chat_generation", __name__)

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"


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
    """
    Count the total number of images in the conversation.
    Returns a tuple of (total_count, list of image messages indices)
    """
    total_images = 0
    image_message_indices = []
    
    for i, message in enumerate(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            image_count = sum(1 for content in message["content"] if isinstance(content, dict) and content.get("type") == "image_url")
            if image_count > 0:
                total_images += image_count
                image_message_indices.append(i)
    
    return total_images, image_message_indices

def manage_conversation_images(messages, new_images_count, engine):
    """
    Manage the conversation to ensure we don't exceed image limits.
    For OpenAI, we maintain only the last 50 images.
    Returns the maximum number of new images we can add.
    """
    if engine != "openai":
        return len(new_images_count)  # No limit for other engines
        
    MAX_IMAGES = 50
    current_total, image_indices = count_images_in_conversation(messages)
    
    # If we already have too many images, remove old image messages
    while current_total > 0 and current_total + new_images_count > MAX_IMAGES and image_indices:
        oldest_image_idx = image_indices[0]
        removed_message = messages.pop(oldest_image_idx)
        # Recount images in the removed message
        removed_images = sum(1 for content in removed_message["content"] 
                           if isinstance(content, dict) and content.get("type") == "image_url")
        current_total -= removed_images
        # Update indices after removal
        image_indices = [idx - 1 for idx in image_indices[1:]]  # Adjust remaining indices
    
    # Return how many new images we can add
    return min(MAX_IMAGES - current_total, new_images_count)

@chat_generation_bp.route("/v1/chat/generation", methods=["POST"])
def generate_code_chat():
    """
    This endpoint generates code for animations using OpenAI or Anthropic.
    It supports both OpenAI and Anthropic models and returns a stream of content.
    
    When calling this endpoint, enable `is_for_platform` to interact with the platform 'Animo'.
    """
    print("Received request for /v1/chat/generation")

    data = request.json
    print(f"Request data: {json.dumps(data, indent=2)}")

    messages = data.get("messages", [])
    prompt = data.get("prompt")
    global_prompt = data.get("globalPrompt", "")
    user_id = data.get("userId") or f"user-{uuid.uuid4()}"
    scenes = data.get("scenes", [])
    project_title = data.get("projectTitle", "")
    engine = data.get("engine", "openai")
    model = data.get("model", None)  # Optional model parameter
    selected_scenes = data.get("selectedScenes", [])
    is_for_platform = data.get("isForPlatform", False)

    # Define default models for each engine
    ENGINE_DEFAULTS = {
        "openai": "gpt-4o",
        "anthropic": "claude-35-sonnet",
        "deepseek": "r1",
        "featherless": "Qwen/Qwen2.5-Coder-7B-Instruct",
    }

    # Validate and set the model based on engine
    if engine not in ENGINE_DEFAULTS:
        return jsonify({"error": f"Invalid engine. Must be one of: {', '.join(ENGINE_DEFAULTS.keys())}"}), 400

    # If no model specified, use the default for the engine
    if not model:
        model = ENGINE_DEFAULTS[engine]

    # Validate model based on engine
    VALID_MODELS = {
        "openai": ["gpt-4o", "o1-mini"],
        "anthropic": ["claude-35-sonnet"],
        "deepseek": ["r1"],
        "featherless": None,
    }

    if VALID_MODELS[engine] is not None and model not in VALID_MODELS[engine]:
        return jsonify({
            "error": f"Invalid model '{model}' for engine '{engine}'. Valid models are: {', '.join(VALID_MODELS[engine])}"
        }), 400

    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]
    
    print("messages")
    print(messages)

    general_system_prompt = """You are an assistant that creates animations with Manim. Manim is a mathematical animation engine that is used to create videos programmatically. You are running on Animo (www.animo.video), a tool to create videos with Manim.

# What the user can do?

The user can create a new project, add scenes, and generate the video. You can help the user to generate the video by creating the code for the scenes. The user can add custom rules for you, can select a different aspect ratio, and can change the model (the models are: OpenAI GPT-4o, and Anthropic Claude 3.5 Sonnet).

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

    if engine == "featherless":
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
                error_message = str(e)
                if is_for_platform:
                    yield f"{json.dumps({'type': 'error', 'text': error_message})}\n"
                else:
                    yield f"Error: {error_message}"

        response = Response(
            stream_with_context(generate()),
            content_type="text/plain; charset=utf-8",
        )
        if is_for_platform:
            response.headers['Transfer-Encoding'] = 'chunked'
            response.headers['x-vercel-ai-data-stream'] = 'v1'
        return response

    if engine == "anthropic":
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def get_preview(code: str, class_name: str):
            """
            get_preview is a function that generates PNGs frames from a Manim script animation.
            
            IMPORTANT: This version of the function will only work for OpenAI models.
            """
            
            print("Generating preview")

            # Get the absolute path of the current script (in api/routes)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_dir = os.path.dirname(current_dir)  # This should be the /api directory
            
            # Create the temporary directory inside /api
            temp_dir = os.path.join(api_dir, "temp_manim")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create the Python file in the temporary location
            file_name = f"{class_name}.py"
            file_path = os.path.join(temp_dir, file_name)
            
            preview_code = f"""
from manim import *
from math import *

{code}
            """
            
            with open(file_path, "w") as f:
                f.write(preview_code)
            
            # Run the Manim command
            command = f"manim {file_path} {class_name} --format=png --media_dir {temp_dir} --custom_folders -pql --disable_caching"
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                
                print(f"Result: {result}")
                
                # Create the previews directory if it doesn't exist
                previews_dir = os.path.join(api_dir, "public", "previews")
                os.makedirs(previews_dir, exist_ok=True)

                # Generate a random string for the subfolder
                random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                
                # Move the generated PNGs to the previews directory
                source_dir = temp_dir
                destination_dir = os.path.join(previews_dir, random_string, class_name)
                
                # Find all PNG files in the source directory
                png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
                
                if png_files:
                    os.makedirs(destination_dir, exist_ok=True)
                    image_list = []
                    for png_file in png_files:
                        shutil.move(os.path.join(source_dir, png_file), os.path.join(destination_dir, png_file))
                        # Extract the index from the filename
                        match = re.search(r'(\d+)\.png$', png_file)
                        if match:
                            index = int(match.group(1))
                            if index % 4 == 0:  # Only include frames where index is divisible by 5
                                image_path = os.path.join(destination_dir, png_file)
                                with Image.open(image_path) as img:
                                    # Calculate new dimensions (half the original size)
                                    width, height = img.size
                                    new_width = width // 4
                                    new_height = height // 4
                                    # Resize the image
                                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                    # Save the resized image to a bytes buffer
                                    buffer = io.BytesIO()
                                    resized_img.save(buffer, format="PNG")
                                    # Get the base64 encoding of the resized image
                                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                image_list.append({
                                    "path": image_path,
                                    "index": index,
                                    "base64": base64_image
                                })
                    image_list.sort(key=lambda x: x["index"])
                    return json.dumps({
                        "message": f"Animation preview generated. Now you will see the image frames in the next automatic message...",
                        "images": image_list
                    })
                else:
                    print(f"No PNG files found in: {source_dir}")
                    return json.dumps({
                        "error": f"No preview files generated at expected location: {source_dir}",
                        "images": []
                    })
            except subprocess.CalledProcessError as e:
                error_output = e.stdout + e.stderr
                print(f"Error running Manim command: {str(e)}")
                print(f"Command output:\n{error_output}")
                return json.dumps({
                    "error": f"ERROR. Error generating preview, please think on what could be the problem, and use `get_preview` to run the code again: {str(e)}\nCommand output:\n{error_output}",
                    "images": []
                })
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return json.dumps({
                    "error": f"Unexpected error: {str(e)}",
                    "images": []
                })

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

        # Extract system message and remove it from the messages list
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
                    print("\n=== Starting new message stream ===")
                    print("=== Current message history ===")
                    for idx, msg in enumerate(messages):
                        print(f"\nMessage {idx}:")
                        print(f"Role: {msg['role']}")
                        if isinstance(msg['content'], list):
                            print("Content (list):")
                            for content_item in msg['content']:
                                if isinstance(content_item, dict):
                                    print(f"  Type: {content_item.get('type', 'unknown')}")
                                    if content_item['type'] == 'text':
                                        print(f"  Text: {content_item['text']}")
                                    elif content_item['type'] == 'tool_result':
                                        print(f"  Tool use ID: {content_item.get('tool_use_id')}")
                        else:
                            print(f"Content: {msg['content']}")
                    print("\n=== End of message history ===")
                    
                    stream = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        messages=messages,
                        system=system_message,
                        max_tokens=1000,
                        stream=True,
                        tools=animo_functions["anthropic"]
                    )
                    
                    current_message = {"role": "assistant", "content": []}
                    current_text = ""  # To accumulate text content
                    json_buffer = ""
                    should_continue = False
                    tool_use_id = None
                    complete_json = ""
                    
                    for chunk in stream:
                        print(f"\nChunk type: {chunk.type}")
                        print(f"Chunk content: {chunk}")
                        
                        if chunk.type == "content_block_start":
                            if hasattr(chunk.content_block, 'type'):
                                if chunk.content_block.type == 'tool_use':
                                    tool_use_id = chunk.content_block.id
                                    print(f"Captured tool_use_id: {tool_use_id}")
                                    
                                    # If we have accumulated text, add it first
                                    if current_text:
                                        current_message["content"].append({
                                            "type": "text",
                                            "text": current_text
                                        })
                                        current_text = ""
                                    
                                    # Add tool use block
                                    tool_input = {}
                                    if complete_json:
                                        try:
                                            tool_input = json.loads(complete_json)
                                        except json.JSONDecodeError:
                                            print("Failed to parse tool input JSON")
                                    
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
                                    print(f"Text content: {content}")
                                    current_text += content  # Accumulate text
                                    if is_for_platform:
                                        for char in content:
                                            escaped_char = repr(char)[1:-1]
                                            yield f'0:"{escaped_char}"\n'
                                    else:
                                        yield content
                                
                            elif hasattr(chunk.delta, 'partial_json'):
                                complete_json += chunk.delta.partial_json
                                print(f"Accumulated complete JSON: {complete_json}")

                        elif chunk.type == "content_block_stop":
                            if complete_json:
                                try:
                                    print("\n=== Processing tool call ===")
                                    tool_call = json.loads(complete_json)
                                    print(f"Tool call data: {json.dumps(tool_call, indent=2)}")
                                    print(f"Using tool_use_id: {tool_use_id}")
                                    
                                    preview_result = get_preview(
                                        code=tool_call.get('code', ''),
                                        class_name=tool_call.get('class_name', '')
                                    )
                                    
                                    try:
                                        preview_data = json.loads(preview_result)
                                        print("\nParsed preview data keys:", preview_data.keys())
                                        
                                        # Just use the middle frame for testing
                                        middle_frame = preview_data['images'][len(preview_data['images'])//2]
                                        base64_data = middle_frame['base64']
                                        
                                        # Debug the base64 data
                                        print(f"\nBase64 data length: {len(base64_data)}")
                                        print(f"Base64 data starts with: {base64_data[:50]}")
                                        print(f"Base64 data ends with: {base64_data[-50:]}")
                                        
                                        # Create simplified content blocks with just one frame
                                        content_blocks = [
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": "image/png",
                                                    "data": base64_data  # Use raw base64 without prefix
                                                }
                                            },
                                            {
                                                "type": "text",
                                                "text": f"\nPreview frame from the animation.\n"
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
                                        
                                        print(f"\nTool response structure:")
                                        print(json.dumps(tool_response, indent=2))
                                        
                                        # Add the assistant's message with tool use before adding tool result
                                        messages.append(current_message)
                                        messages.append(tool_response)
                                        should_continue = True
                                        
                                        preview_text = "Generated preview of the animation:\n"
                                        if is_for_platform:
                                            for char in preview_text:
                                                escaped_char = repr(char)[1:-1]
                                                yield f'0:"{escaped_char}"\n'
                                            yield f'0:"[IMAGE: Preview frame]"\n'
                                        else:
                                            yield "\n[Preview frame]\n"
                                        
                                    except json.JSONDecodeError:
                                        print("Failed to parse preview result as JSON")
                                        print("Raw preview result:", preview_result)
                                        continue
                                
                                except Exception as e:
                                    print(f"Error processing tool call: {str(e)}")
                                    continue
                        
                        elif chunk.type == "message_stop":
                            print("\n=== Message stream ended ===")
                            # Add any remaining text content
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
                print(f"\n=== Error occurred ===\nError details: {str(e)}")
                error_message = f'0:"{str(e)}"\n' if is_for_platform else f"Error: {str(e)}"
                yield error_message

        response = Response(stream_with_context(generate()), content_type="text/plain; charset=utf-8" if is_for_platform else "text/event-stream")
        if is_for_platform:
            response.headers['Transfer-Encoding'] = 'chunked'
            response.headers['x-vercel-ai-data-stream'] = 'v1'
        return response

    else:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
        def get_preview(code: str, class_name: str):
            """
            get_preview is a function that generates PNGs frames from a Manim script animation.
            
            IMPORTANT: This version of the function will only work for OpenAI models.
            """
            
            print("Generating preview")

            # Get the absolute path of the current script (in api/routes)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_dir = os.path.dirname(current_dir)  # This should be the /api directory
            
            # Create the temporary directory inside /api
            temp_dir = os.path.join(api_dir, "temp_manim")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create the Python file in the temporary location
            file_name = f"{class_name}.py"
            file_path = os.path.join(temp_dir, file_name)
            
            preview_code = f"""
from manim import *
from math import *

{code}
            """
            
            with open(file_path, "w") as f:
                f.write(preview_code)
            
            # Run the Manim command
            command = f"manim {file_path} {class_name} --format=png --media_dir {temp_dir} --custom_folders -pql --disable_caching"
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                
                print(f"Result: {result}")
                
                # Create the previews directory if it doesn't exist
                previews_dir = os.path.join(api_dir, "public", "previews")
                os.makedirs(previews_dir, exist_ok=True)

                # Generate a random string for the subfolder
                random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                
                # Move the generated PNGs to the previews directory
                source_dir = temp_dir
                destination_dir = os.path.join(previews_dir, random_string, class_name)
                
                # Find all PNG files in the source directory
                png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
                
                if png_files:
                    os.makedirs(destination_dir, exist_ok=True)
                    image_list = []
                    for png_file in png_files:
                        shutil.move(os.path.join(source_dir, png_file), os.path.join(destination_dir, png_file))
                        # Extract the index from the filename
                        match = re.search(r'(\d+)\.png$', png_file)
                        if match:
                            index = int(match.group(1))
                            if index % 4 == 0:  # Only include frames where index is divisible by 5
                                image_path = os.path.join(destination_dir, png_file)
                                with Image.open(image_path) as img:
                                    # Calculate new dimensions (half the original size)
                                    width, height = img.size
                                    new_width = width // 4
                                    new_height = height // 4
                                    # Resize the image
                                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                    # Save the resized image to a bytes buffer
                                    buffer = io.BytesIO()
                                    resized_img.save(buffer, format="PNG")
                                    # Get the base64 encoding of the resized image
                                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                image_list.append({
                                    "path": image_path,
                                    "index": index,
                                    "base64": base64_image
                                })
                    image_list.sort(key=lambda x: x["index"])
                    return json.dumps({
                        "message": f"Animation preview generated. Now you will see the image frames in the next automatic message...",
                        "images": image_list
                    })
                else:
                    print(f"No PNG files found in: {source_dir}")
                    return json.dumps({
                        "error": f"No preview files generated at expected location: {source_dir}",
                        "images": []
                    })
            except subprocess.CalledProcessError as e:
                error_output = e.stdout + e.stderr
                print(f"Error running Manim command: {str(e)}")
                print(f"Command output:\n{error_output}")
                return json.dumps({
                    "error": f"ERROR. Error generating preview, please think on what could be the problem, and use `get_preview` to run the code again: {str(e)}\nCommand output:\n{error_output}",
                    "images": []
                })
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return json.dumps({
                    "error": f"Unexpected error: {str(e)}",
                    "images": []
                })

        def generate():
            max_retries = 3
            retry_delay = 4  # seconds

            while True:
                for attempt in range(max_retries):
                    try:
                        stream = client.chat.completions.create(
                            model="gpt-4o",
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
                        
                        # If we get here, the stream completed successfully
                        break
                    
                    except APIError as e:
                        if attempt < max_retries - 1:
                            print(f"APIError occurred: {str(e)}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"Max retries reached. APIError: {str(e)}")
                            yield json.dumps({"error": "Max retries reached due to API errors"})
                            return  # Exit the generator

                if function_call_data:
                    # Add the function call to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": function_call_data
                        }
                    })

                    # Yield the whole object back to the frontend
                    function_call_obj = json.dumps({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": function_call_data
                        }
                    })
                    if is_for_platform:
                        pass
                        # text_obj = json.dumps({"type": "text", "text": function_call_obj})
                        # yield f'{text_obj}\n'
                    else:
                        pass

                    # Actually call get_preview
                    if function_name == "get_preview":
                        print(f"Calling get_preview with data: {function_call_data}")
                        args = json.loads(function_call_data)
                        result = get_preview(args['code'], args['class_name'])
                        result_json = json.loads(result)
                        function_response = {
                            "content": result_json.get("message", result_json.get("error")),
                            "name": "get_preview",
                            "role": "function"
                        }
                        messages.append(function_response)

                        # Yield the function response back to the frontend
                        if is_for_platform:
                            function_result_obj = json.dumps({
                                "type": "function_result",
                                "content": function_response,
                                "function_call": {"name": function_name}
                            })
                            yield f'{function_result_obj}\n'
                        else:
                            pass

                        # Only create and send image_message if there are images
                        if result_json.get("images"):
                            # Create a new message with the images
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

                            # Calculate how many new images we can add
                            available_slots = manage_conversation_images(messages, len(result_json["images"]), engine)
                            
                            # Select frames based on available slots
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

                            # Yield the image message back to the frontend
                            image_message_obj = json.dumps(image_message)
                            if is_for_platform:
                                pass
                                # text_obj = json.dumps({"type": "text", "text": image_message_obj})
                                # yield f'{text_obj}\n'
                            else:
                                yield image_message_obj

                        # Trigger a new response from the assistant
                        continue  # This will start a new iteration of the while loop
                    else:
                        break  # Exit the loop if it's not a get_preview function call
                else:
                    break  # Exit the loop if there's no function call

            # Final message when there are no more function calls
            final_message = "\n"
            if is_for_platform:
                text_obj = json.dumps({"type": "text", "text": final_message})
                yield f'{text_obj}\n'
            else:
                yield final_message

        print("Generating response")
        response = Response(stream_with_context(generate()), content_type="text/plain; charset=utf-8")
        if is_for_platform:
            response.headers['Transfer-Encoding'] = 'chunked'
            response.headers['x-vercel-ai-data-stream'] = 'v1'
        return response
