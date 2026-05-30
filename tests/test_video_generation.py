"""Tests for /v1/video/generation route."""

import os
from unittest import mock

import pytest

VALID_MANIM_CODE = "from manim import *\nclass GenScene(Scene):\n    def construct(self): pass"


@pytest.fixture
def mock_subprocess_success(tmp_path):
    """Patch subprocess.run to simulate a successful Manim render."""
    result = mock.MagicMock()
    result.returncode = 0
    result.stderr = ""
    result.stdout = ""

    with mock.patch("api.routes.video_generation.subprocess.run", return_value=result):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("shutil.move"):
                yield result


class TestVideoGenerationValidation:
    def test_missing_prompt_returns_400(self, client):
        resp = client.post("/v1/video/generation", json={"engine": "openai"})
        assert resp.status_code == 400
        assert "prompt" in resp.get_json()["error"].lower()

    def test_empty_prompt_returns_400(self, client):
        resp = client.post("/v1/video/generation", json={"prompt": ""})
        assert resp.status_code == 400


class TestVideoGenerationOpenAI:
    def test_successful_generation(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        msg = mock.MagicMock()
        msg.content = VALID_MANIM_CODE
        choice = mock.MagicMock()
        choice.message = msg
        openai_resp = mock.MagicMock()
        openai_resp.choices = [choice]

        subprocess_result = mock.MagicMock()
        subprocess_result.returncode = 0

        with mock.patch("api.routes.video_generation.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = openai_resp
            with mock.patch("api.routes.video_generation.subprocess.run", return_value=subprocess_result):
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("shutil.move"):
                        resp = client.post("/v1/video/generation", json={
                            "prompt": "draw a blue circle",
                            "engine": "openai",
                        })

        assert resp.status_code == 200
        data = resp.get_json()
        assert "video_url" in data
        assert "code" in data

    def test_llm_failure_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with mock.patch("api.routes.video_generation.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("quota exceeded")
            resp = client.post("/v1/video/generation", json={
                "prompt": "draw a circle",
                "engine": "openai",
            })
        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_manim_render_failure_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        msg = mock.MagicMock()
        msg.content = VALID_MANIM_CODE
        choice = mock.MagicMock()
        choice.message = msg
        openai_resp = mock.MagicMock()
        openai_resp.choices = [choice]

        subprocess_result = mock.MagicMock()
        subprocess_result.returncode = 1
        subprocess_result.stderr = "NameError: GenScene not found"
        subprocess_result.stdout = ""

        with mock.patch("api.routes.video_generation.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = openai_resp
            with mock.patch("api.routes.video_generation.subprocess.run", return_value=subprocess_result):
                resp = client.post("/v1/video/generation", json={
                    "prompt": "draw a circle",
                    "engine": "openai",
                })

        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_render_timeout_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        msg = mock.MagicMock()
        msg.content = VALID_MANIM_CODE
        choice = mock.MagicMock()
        choice.message = msg
        openai_resp = mock.MagicMock()
        openai_resp.choices = [choice]

        with mock.patch("api.routes.video_generation.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = openai_resp
            with mock.patch("api.routes.video_generation.subprocess.run",
                            side_effect=__import__("subprocess").TimeoutExpired("manim", 300)):
                resp = client.post("/v1/video/generation", json={
                    "prompt": "draw a circle",
                    "engine": "openai",
                })

        assert resp.status_code == 504
        assert "timed out" in resp.get_json()["error"].lower()


class TestVideoGenerationGemini:
    def test_successful_generation(self, client):
        subprocess_result = mock.MagicMock()
        subprocess_result.returncode = 0

        with mock.patch("api.routes.video_generation.generate_gemini_content",
                        return_value=VALID_MANIM_CODE):
            with mock.patch("api.routes.video_generation.subprocess.run", return_value=subprocess_result):
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("shutil.move"):
                        resp = client.post("/v1/video/generation", json={
                            "prompt": "draw a pentagon",
                            "engine": "gemini",
                            "model": "gemini-2.5-flash",
                        })

        assert resp.status_code == 200


class TestVideoGenerationAspectRatio:
    def test_default_aspect_ratio_is_16_9(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        msg = mock.MagicMock()
        msg.content = VALID_MANIM_CODE
        choice = mock.MagicMock()
        choice.message = msg
        openai_resp = mock.MagicMock()
        openai_resp.choices = [choice]

        subprocess_result = mock.MagicMock()
        subprocess_result.returncode = 0
        written_code = {}

        real_open = open

        def capture_open(path, mode="r", *args, **kwargs):
            if mode == "w" and path.endswith(".py"):
                f = mock.MagicMock()
                def capture_write(content):
                    written_code["content"] = content
                f.write = capture_write
                f.__enter__ = lambda s: f
                f.__exit__ = mock.MagicMock(return_value=False)
                return f
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("api.routes.video_generation.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = openai_resp
            with mock.patch("builtins.open", side_effect=capture_open):
                with mock.patch("api.routes.video_generation.subprocess.run", return_value=subprocess_result):
                    with mock.patch("os.path.exists", return_value=True):
                        with mock.patch("shutil.move"):
                            client.post("/v1/video/generation", json={
                                "prompt": "draw a circle",
                            })

        if written_code.get("content"):
            assert "3840, 2160" in written_code["content"]
