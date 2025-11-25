from flask import Blueprint, jsonify, current_app, request, Response
import subprocess
import os
import re
import json
import sys
import traceback
from azure.storage.blob import BlobServiceClient
import shutil
from typing import Union
import uuid
import time
import requests

video_rendering_bp = Blueprint("video_rendering", __name__)


USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "true") == "true"
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8080")


def upload_to_azure_storage(file_path: str, video_storage_file_name: str) -> str:
    """
    Uploads the video to Azure Blob Storage and returns the URL.
    """
    cloud_file_name = f"{video_storage_file_name}.mp4"

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=cloud_file_name
    )

    # Upload the video file
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    # Construct the URL of the uploaded blob
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{cloud_file_name}"
    return blob_url


def move_to_public_folder(
    file_path: str, video_storage_file_name: str, base_url: Union[str, None] = None
) -> str:
    """
    Moves the video to the public folder and returns the URL.
    """
    # Get the api directory (go up one level from routes)
    api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    public_folder = os.path.join(api_dir, "public")
    os.makedirs(public_folder, exist_ok=True)

    new_file_name = f"{video_storage_file_name}.mp4"
    new_file_path = os.path.join(public_folder, new_file_name)

    shutil.move(file_path, new_file_path)

    # Use the provided base_url if available, otherwise fall back to BASE_URL
    url_base = base_url if base_url else BASE_URL
    video_url = f"{url_base.rstrip('/')}/public/{new_file_name}"
    return video_url


def get_frame_config(aspect_ratio):
    if aspect_ratio == "16:9":
        return (3840, 2160), 14.22
    elif aspect_ratio == "9:16":
        return (1080, 1920), 8.0
    elif aspect_ratio == "1:1":
        return (1080, 1080), 8.0
    else:
        return (3840, 2160), 14.22


@video_rendering_bp.route("/v1/video/rendering", methods=["POST"])
def render_video():
    # Get the API key from the request headers
    # api_key = request.headers.get('X-API-Key')
    
    # if not api_key:
    #     return jsonify({"error": "API key is missing"}), 401
    
    # Validate the API key and get the user ID
    # user_id = get_user_by_api_key(api_key)
    
    # if not user_id:
    #     return jsonify({"error": "Invalid API key"}), 401
    
    # Now that we have a valid user_id, create a run
    # run_id = create_run_on_user(user_id, "video")
    
    # Extract the rest of the request data
    code = request.json.get("code")
    file_name = request.json.get("file_name")
    file_class = request.json.get("file_class")

    user_id = request.json.get("user_id") or str(uuid.uuid4())
    project_name = request.json.get("project_name")
    iteration = request.json.get("iteration")

    # Aspect Ratio can be: "16:9" (default), "1:1", "9:16"
    aspect_ratio = request.json.get("aspect_ratio")

    # Stream the percentage of animation it shown in the error
    stream = request.json.get("stream", False)

    video_storage_file_name = f"video-{user_id}-{project_name}-{iteration}"

    if not code:
        return jsonify(error="No code provided"), 400

    # Determine frame size and width based on aspect ratio
    frame_size, frame_width = get_frame_config(aspect_ratio)

    # Modify the Manim script to include configuration settings
    modified_code = f"""
from manim import *
from math import *
config.frame_size = {frame_size}
config.frame_width = {frame_width}

{code}
    """

    # Create a unique file name
    file_name = f"scene_{os.urandom(2).hex()}.py"
    
    # Adjust the path to point to /api/public/
    api_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from routes
    public_dir = os.path.join(api_dir, "public")
    os.makedirs(public_dir, exist_ok=True)  # Ensure the public directory exists
    file_path = os.path.join(public_dir, file_name)

    # Write the code to the file
    with open(file_path, "w") as f:
        f.write(modified_code)

    def render_video():
        try:
            command_list = [
                "manim",
                file_path,  # Use the full path to the file
                file_class,
                "--format=mp4",
                "--media_dir",
                ".",
                "--custom_folders",
            ]

            process = subprocess.Popen(
                command_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.realpath(__file__)),
                text=True,
                bufsize=1,  # Ensure the output is in text mode and line-buffered
            )
            current_animation = -1
            current_percentage = 0
            error_output = []
            in_error = False

            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output == "" and error == "" and process.poll() is not None:
                    break

                if output:
                    print("STDOUT:", output.strip())
                if error:
                    print("STDERR:", error.strip())
                    error_output.append(error.strip())
                    
                # Check for critical errors
                if "is not in the script" in error:
                    in_error = True
                    continue

                # Check for start of error
                if "Traceback (most recent call last)" in error:
                    in_error = True
                    continue

                # If we're in an error state, keep accumulating the error message
                if in_error:
                    if error.strip() == "":
                        # Empty line might indicate end of traceback
                        in_error = False
                        full_error = "\n".join(error_output)
                        yield f'{{"error": {json.dumps(full_error)}}}\n'
                        return
                    continue

                animation_match = re.search(r"Animation (\d+):", error)
                if animation_match:
                    new_animation = int(animation_match.group(1))
                    if new_animation != current_animation:
                        current_animation = new_animation
                        current_percentage = 0
                        yield f'{{"animationIndex": {current_animation}, "percentage": 0}}\n'

                percentage_match = re.search(r"(\d+)%", error)
                if percentage_match:
                    new_percentage = int(percentage_match.group(1))
                    if new_percentage != current_percentage:
                        current_percentage = new_percentage
                        yield f'{{"animationIndex": {current_animation}, "percentage": {current_percentage}}}\n'

            if process.returncode == 0:
                # Update this part
                video_file_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    f"{file_class or 'GenScene'}.mp4"
                )
                # Looking for video file at: {video_file_path}
                
                if not os.path.exists(video_file_path):
                    #  Video file not found. Searching in parent directory...
                    video_file_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        f"{file_class or 'GenScene'}.mp4"
                    )
                    # New video file path is: {video_file_path}

                if os.path.exists(video_file_path):
                    print(f"Video file found at: {video_file_path}")
                else:
                    print(f"Video file not found. Files in current directory: {os.listdir(os.path.dirname(video_file_path))}")
                    raise FileNotFoundError(f"Video file not found at {video_file_path}")

                print(f"Files in video file directory: {os.listdir(os.path.dirname(video_file_path))}")
                
                if USE_LOCAL_STORAGE:
                    # Pass request.host_url if available
                    base_url = (
                        request.host_url
                        if request and hasattr(request, "host_url")
                        else None
                    )
                    video_url = move_to_public_folder(
                        video_file_path, video_storage_file_name, base_url
                    )
                else:
                    video_url = upload_to_azure_storage(
                        video_file_path, video_storage_file_name
                    )
                print(f"Video URL: {video_url}")
                if stream:
                    yield f'{{ "video_url": "{video_url}" }}\n'
                    sys.stdout.flush()
                else:
                    yield json.dumps({
                        "message": "Video generation completed",
                        "video_url": video_url,
                    })
            else:
                full_error = "\n".join(error_output)
                yield f'{{"error": {json.dumps(full_error)}}}\n'

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            print(f"Files in current directory after error: {os.listdir('.')}")
            yield f'{{"error": "Unexpected error occurred: {str(e)}"}}\n'
        finally:
            # Remove the temporary Python file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed temporary file: {file_path}")
                # Remove the video file
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)
                    print(f"Removed temporary video file: {video_file_path}")
            except Exception as e:
                print(f"Error removing temporary file {file_path}: {e}")

    if stream:
        # TODO: If the `render_video()` fails, or it's sending {"error"}, be sure to add `500`
        return Response(
            render_video(), content_type="text/event-stream", status=207
        )
    else:
        video_url = None
        try:
            for result in render_video():  # Iterate through the generator
                print(f"Generated result: {result}")  # Debug print
                try:
                    result_dict = json.loads(result)
                    if "video_url" in result_dict:
                        video_url = result_dict["video_url"]
                    elif "error" in result_dict:
                        raise Exception(result_dict["error"])
                except (json.JSONDecodeError, TypeError):
                    # If it's not a JSON string, try treating it as a dict (shouldn't happen now)
                    if isinstance(result, dict):
                        if "video_url" in result:
                            video_url = result["video_url"]
                        elif "error" in result:
                            raise Exception(result["error"])

            if video_url:
                return (
                    jsonify(
                        {
                            "message": "Video generation completed",
                            "video_url": video_url,
                        }
                    ),
                    200,
                )
            else:
                return (
                    jsonify(
                        {
                            "message": "Video generation completed, but no URL was found"
                        }
                    ),
                    200,
                )
        except StopIteration:
            if video_url:
                return (
                    jsonify(
                        {
                            "message": "Video generation completed",
                            "video_url": video_url,
                        }
                    ),
                    200,
                )
            else:
                return (
                    jsonify(
                        {
                            "message": "Video generation completed, but no URL was found"
                        }
                    ),
                    200,
                )
        except Exception as e:
            print(f"Error in non-streaming mode: {e}")
            return jsonify({"error": str(e)}), 500


@video_rendering_bp.route("/v1/video/exporting", methods=["POST"])
def export_video():
    scenes = request.json.get("scenes")
    title_slug = request.json.get("titleSlug")
    local_filenames = []

    # Download each scene
    for scene in scenes:
        video_url = scene["videoUrl"]
        object_name = video_url.split("/")[-1]
        local_filename = download_video(video_url)
        local_filenames.append(local_filename)

    # Create a list of input file arguments for ffmpeg
    input_files = " ".join([f"-i {filename}" for filename in local_filenames])

    # Generate a unique filename with UNIX timestamp
    timestamp = int(time.time())
    merged_filename = os.path.join(
        os.getcwd(), f"exported-scene-{title_slug}-{timestamp}.mp4"
    )

    # Command to merge videos using ffmpeg
    command = f"ffmpeg {input_files} -filter_complex 'concat=n={len(local_filenames)}:v=1:a=0[out]' -map '[out]' {merged_filename}"

    try:
        # Execute the ffmpeg command
        subprocess.run(command, shell=True, check=True)
        print("Videos merged successfully.")
        print(f"merged_filename: {merged_filename}")
        public_url = upload_to_azure_storage(
            merged_filename, f"exported-scene-{title_slug}-{timestamp}"
        )
        print(f"Video URL: {public_url}")
        return jsonify(
            {"status": "Videos merged successfully", "video_url": public_url}
        )
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e}")
        return jsonify({"error": "Failed to merge videos"}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


def download_video(video_url):
    local_filename = video_url.split("/")[-1]
    response = requests.get(video_url)
    response.raise_for_status()
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    return local_filename
