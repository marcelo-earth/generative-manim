"""Tests for /v1/code/generation route (OpenAI, Anthropic, Gemini engines)."""

from unittest import mock

import pytest


def _make_openai_response(content):
    msg = mock.MagicMock()
    msg.content = content
    choice = mock.MagicMock()
    choice.message = msg
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


def _make_anthropic_response(content):
    block = mock.MagicMock()
    block.text = content
    resp = mock.MagicMock()
    resp.content = [block]
    return resp


class TestCodeGenerationInvalidInput:
    def test_invalid_engine_returns_400(self, client):
        resp = client.post("/v1/code/generation", json={
            "prompt": "draw a circle",
            "engine": "invalid_engine",
        })
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_missing_prompt_defaults_to_empty_string(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _make_openai_response("from manim import *")
            )
            resp = client.post("/v1/code/generation", json={"engine": "openai"})
        assert resp.status_code == 200


class TestCodeGenerationOpenAI:
    def test_successful_response(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _make_openai_response("from manim import *\nclass GenScene(Scene): pass")
            )
            resp = client.post("/v1/code/generation", json={
                "prompt": "draw a circle",
                "engine": "openai",
            })
        assert resp.status_code == 200
        assert "code" in resp.get_json()

    def test_uses_default_model_gpt4o(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured = {}
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            def capture(**kwargs):
                captured.update(kwargs)
                return _make_openai_response("code")
            mock_client.return_value.chat.completions.create.side_effect = capture
            client.post("/v1/code/generation", json={"prompt": "test", "engine": "openai"})
        assert captured.get("model") == "gpt-4o"

    def test_custom_model_forwarded(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured = {}
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            def capture(**kwargs):
                captured.update(kwargs)
                return _make_openai_response("code")
            mock_client.return_value.chat.completions.create.side_effect = capture
            client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "openai",
                "model": "gpt-4-turbo",
            })
        assert captured.get("model") == "gpt-4-turbo"

    def test_api_error_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = RuntimeError("API down")
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "openai",
            })
        assert resp.status_code == 500

    def test_missing_api_key_raises_500(self, client, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            mock_client.side_effect = ValueError("OPENAI_API_KEY is required")
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "openai",
            })
        assert resp.status_code == 500


class TestCodeGenerationAnthropic:
    def test_successful_response(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = (
                _make_anthropic_response("from manim import *")
            )
            resp = client.post("/v1/code/generation", json={
                "prompt": "draw a square",
                "engine": "anthropic",
            })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["code"] == "from manim import *"

    def test_uses_default_claude_sonnet_4_6(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        captured = {}
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            def capture(**kwargs):
                captured.update(kwargs)
                return _make_anthropic_response("code")
            mock_anthropic.return_value.messages.create.side_effect = capture
            client.post("/v1/code/generation", json={"prompt": "test", "engine": "anthropic"})
        assert captured.get("model") == "claude-sonnet-4-6"

    def test_claude_opus_4_7_accepted(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        captured = {}
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            def capture(**kwargs):
                captured.update(kwargs)
                return _make_anthropic_response("code")
            mock_anthropic.return_value.messages.create.side_effect = capture
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "anthropic",
                "model": "claude-opus-4-7",
            })
        assert resp.status_code == 200
        assert captured.get("model") == "claude-opus-4-7"

    def test_multiple_content_blocks_joined(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        block1 = mock.MagicMock()
        block1.text = "from manim import *\n"
        block2 = mock.MagicMock()
        block2.text = "class GenScene(Scene): pass"
        resp_obj = mock.MagicMock()
        resp_obj.content = [block1, block2]
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = resp_obj
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "anthropic",
            })
        assert resp.status_code == 200
        assert "from manim import *" in resp.get_json()["code"]
        assert "GenScene" in resp.get_json()["code"]

    def test_api_error_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = RuntimeError("overloaded")
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "anthropic",
            })
        assert resp.status_code == 500


class TestCodeGenerationGemini:
    def test_successful_response(self, client, monkeypatch):
        with mock.patch("api.routes.code_generation.generate_gemini_content") as mock_gemini:
            mock_gemini.return_value = "from manim import *"
            resp = client.post("/v1/code/generation", json={
                "prompt": "draw a triangle",
                "engine": "gemini",
            })
        assert resp.status_code == 200
        assert resp.get_json()["code"] == "from manim import *"

    def test_error_returns_500(self, client, monkeypatch):
        with mock.patch("api.routes.code_generation.generate_gemini_content") as mock_gemini:
            mock_gemini.side_effect = ValueError("GEMINI_API_KEY is required")
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "gemini",
            })
        assert resp.status_code == 500

    def test_default_model_forwarded(self, client, monkeypatch):
        with mock.patch("api.routes.code_generation.generate_gemini_content") as mock_gemini:
            mock_gemini.return_value = "code"
            client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "gemini",
            })
        args = mock_gemini.call_args[0]
        assert args[0] == "gemini-2.5-flash"


class TestCodeGenerationFeatherless:
    def test_successful_response(self, client, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _make_openai_response("from manim import *")
            )
            resp = client.post("/v1/code/generation", json={
                "prompt": "draw a hexagon",
                "engine": "featherless",
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            })
        assert resp.status_code == 200

    def test_missing_api_key_returns_500(self, client, monkeypatch):
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        with mock.patch("api.routes.code_generation.get_openai_compatible_client") as mock_client:
            mock_client.side_effect = ValueError("FEATHERLESS_API_KEY is required")
            resp = client.post("/v1/code/generation", json={
                "prompt": "test",
                "engine": "featherless",
            })
        assert resp.status_code == 500
