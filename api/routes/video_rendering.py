import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from typing import Union

import requests
from azure.storage.blob import BlobServiceClient
from flask import Blueprint, Response, jsonify, request

from api.validation import get_json_body, require_string, validate_aspect_ratio, validate_boolean
from api.video_utils import assert_public_http_url, get_frame_config

video_rendering_bp = Blueprint("video_rendering", __name__)


USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "true") == "true"
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8080")


def upload_to_azure_storage(file_path: str, video_storage_file_name: str) -> str:
    cloud_file_name = f"{video_storage_file_name}.mp4"

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=cloud_file_name
    )

    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{cloud_file_name}"
    return blob_url


def move_to_public_folder(
    file_path: str, video_storage_file_name: str, base_url: Union[str, None] = None
) -> str:
    api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    public_folder = os.path.join(api_dir, "public")
    os.makedirs(public_folder, exist_ok=True)

    new_file_name = f"{video_storage_file_name}.mp4"
    new_file_path = os.path.join(public_folder, new_file_name)

    shutil.move(file_path, new_file_path)

    url_base = base_url if base_url else BASE_URL
    video_url = f"{url_base.rstrip('/')}/public/{new_file_name}"
    return video_url


@video_rendering_bp.route("/v1/video/rendering", methods=["POST"])
def render_video():
    body, err = get_json_body()
    if err:
        return err

    code, err = require_string(body, "code")
    if err:
        return err

    aspect_ratio, err = validate_aspect_ratio(body)
    if err:
        return err

    stream, err = validate_boolean(body, "stream", default=False)
    if err:
        return err

    file_name = body.get("file_name")
    file_class = body.get("file_class")
    user_id = body.get("user_id") or str(uuid.uuid4())
    project_name = body.get("project_name")
    iteration = body.get("iteration")

    video_storage_file_name = f"video-{user_id}-{project_name}-{iteration}"

    frame_size, frame_width = get_frame_config(aspect_ratio)

    modified_code = f"""
from manim import *
from math import *
config.frame_size = {frame_size}
config.frame_width = {frame_width}

{code}
    """

    file_name = f"scene_{os.urandom(2).hex()}.py"
    api_dir = os.path.dirname(os.path.dirname(__file__))
    public_dir = os.path.join(api_dir, "public")
    os.makedirs(public_dir, exist_ok=True)
    file_path = os.path.join(public_dir, file_name)

    with open(file_path, "w") as f:
        f.write(modified_code)

    def render_video():
        try:
            command_list = [
                "manim",
                file_path,
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
                bufsize=1,
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

                if error:
                    error_output.append(error.strip())

                if "is not in the script" in error:
                    in_error = True
                    continue

                if "Traceback (most recent call last)" in error:
                    in_error = True
                    continue

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
                video_file_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    f"{file_class or 'GenScene'}.mp4"
                )
                if not os.path.exists(video_file_path):
                    video_file_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        f"{file_class or 'GenScene'}.mp4"
                    )
                if not os.path.exists(video_file_path):
                    raise FileNotFoundError(f"Video file not found at {video_file_path}")

                if USE_LOCAL_STORAGE:
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
            traceback.print_exc()
            yield f'{{"error": "Unexpected error occurred: {str(e)}"}}\n'
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)
            except Exception:
                pass

    if stream:
        return Response(
            render_video(), content_type="text/event-stream", status=207
        )
    else:
        video_url = None
        try:
            for result in render_video():
                try:
                    result_dict = json.loads(result)
                    if "video_url" in result_dict:
                        video_url = result_dict["video_url"]
                    elif "error" in result_dict:
                        raise Exception(result_dict["error"])
                except (json.JSONDecodeError, TypeError):
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
            return jsonify({"error": str(e)}), 500


@video_rendering_bp.route("/v1/video/exporting", methods=["POST"])
def export_video():
    body, err = get_json_body()
    if err:
        return err

    scenes = body.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return jsonify({"error": "'scenes' must be a non-empty array"}), 400

    title_slug = body.get("titleSlug")
    local_filenames = []

    try:
        for scene in scenes:
            video_url = scene.get("videoUrl") if isinstance(scene, dict) else None
            local_filename = download_video(video_url)
            local_filenames.append(local_filename)
    except ValueError as e:
        return jsonify({"error": f"Invalid videoUrl: {e}"}), 400
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download scene video: {e}"}), 400

    timestamp = int(time.time())
    safe_slug = re.sub(r'[^a-zA-Z0-9_-]', '', title_slug or 'untitled')
    merged_filename = os.path.join(
        os.getcwd(), f"exported-scene-{safe_slug}-{timestamp}.mp4"
    )

    command_list = ["ffmpeg"]
    for filename in local_filenames:
        command_list.extend(["-i", filename])
    command_list.extend([
        "-filter_complex", f"concat=n={len(local_filenames)}:v=1:a=0[out]",
        "-map", "[out]",
        merged_filename
    ])

    try:
        subprocess.run(command_list, check=True)
        public_url = upload_to_azure_storage(
            merged_filename, f"exported-scene-{safe_slug}-{timestamp}"
        )
        return jsonify({"status": "Videos merged successfully", "video_url": public_url})
    except subprocess.CalledProcessError:
        return jsonify({"error": "Failed to merge videos"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for filename in local_filenames:
            try:
                os.remove(filename)
            except OSError:
                pass


def download_video(video_url):
    assert_public_http_url(video_url)

    local_filename = os.path.join(os.getcwd(), f"scene-download-{uuid.uuid4().hex}.mp4")
    with requests.get(video_url, stream=True, timeout=30, allow_redirects=False) as response:
        if 300 <= response.status_code < 400:
            raise ValueError("redirects are not allowed for scene video URLs")
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return local_filename
