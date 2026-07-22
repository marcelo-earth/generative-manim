"""Tests for /v1/chat/generation route validation and engine routing."""

from unittest import mock

import pytest


class TestChatGenerationEngineValidation:
    def test_invalid_engine_returns_400(self, client):
        resp = client.post("/v1/chat/generation", json={
            "prompt": "draw a circle",
            "engine": "unknown_engine",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "engine" in data["error"].lower()

    def test_invalid_model_for_openai_returns_400(self, client):
        resp = client.post("/v1/chat/generation", json={
            "prompt": "draw a circle",
            "engine": "openai",
            "model": "gpt-99-turbo",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "gpt-99-turbo" in data["error"]

    def test_invalid_model_for_anthropic_returns_400(self, client):
        resp = client.post("/v1/chat/generation", json={
            "prompt": "draw a circle",
            "engine": "anthropic",
            "model": "claude-fake-9",
        })
        assert resp.status_code == 400

    def test_invalid_model_for_deepseek_returns_400(self, client):
        resp = client.post("/v1/chat/generation", json={
            "prompt": "draw a circle",
            "engine": "deepseek",
            "model": "r2",
        })
        assert resp.status_code == 400

    def test_valid_openai_model_passes_validation(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_stream = mock.MagicMock()
            mock_stream.__iter__ = mock.Mock(return_value=iter([]))
            mock_openai.return_value.chat.completions.create.return_value = mock_stream
            resp = client.post("/v1/chat/generation", json={
                "prompt": "draw a circle",
                "engine": "openai",
                "model": "o1-mini",
            })
        assert resp.status_code == 200

    def test_valid_anthropic_model_passes_validation(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_stream = mock.MagicMock()
            mock_stream.__iter__ = mock.Mock(return_value=iter([]))
            mock_anthropic.return_value.messages.create.return_value = mock_stream
            resp = client.post("/v1/chat/generation", json={
                "prompt": "draw a circle",
                "engine": "anthropic",
                "model": "claude-sonnet-4-6",
            })
        assert resp.status_code == 200

    def test_featherless_accepts_any_model(self, client, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_stream = mock.MagicMock()
            mock_stream.__iter__ = mock.Mock(return_value=iter([]))
            mock_openai.return_value.chat.completions.create.return_value = mock_stream
            resp = client.post("/v1/chat/generation", json={
                "prompt": "draw a circle",
                "engine": "featherless",
                "model": "any-open-weight-model",
            })
        assert resp.status_code == 200

    def test_litellm_accepts_any_model(self, client):
        import sys
        litellm_stub = sys.modules["litellm"]
        litellm_stub.completion.return_value = iter([])
        resp = client.post("/v1/chat/generation", json={
            "prompt": "draw a circle",
            "engine": "litellm",
            "model": "groq/llama-3.3-70b",
        })
        assert resp.status_code == 200


class TestChatGenerationModelDefaults:
    def test_openai_default_model_is_gpt4o(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured = {}
        with mock.patch("openai.OpenAI") as mock_openai:
            def capture(**kwargs):
                captured.update(kwargs)
                return iter([])
            mock_openai.return_value.chat.completions.create.side_effect = capture
            client.post("/v1/chat/generation", json={"prompt": "test", "engine": "openai"})
        assert captured.get("model") == "gpt-4o"

    def test_anthropic_default_model_is_claude_sonnet_5(self, client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        captured = {}
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            def capture(**kwargs):
                captured.update(kwargs)
                return iter([])
            mock_anthropic.return_value.messages.create.side_effect = capture
            client.post("/v1/chat/generation", json={"prompt": "test", "engine": "anthropic"})
        assert captured.get("model") == "claude-sonnet-5"


class TestChatGenerationPromptMessages:
    def test_prompt_converted_to_messages(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured = {}
        with mock.patch("openai.OpenAI") as mock_openai:
            def capture(**kwargs):
                captured.update(kwargs)
                return iter([])
            mock_openai.return_value.chat.completions.create.side_effect = capture
            client.post("/v1/chat/generation", json={
                "prompt": "draw a blue circle",
                "engine": "openai",
            })
        messages = captured.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any("blue circle" in str(m.get("content", "")) for m in user_messages)

    def test_empty_body_returns_400(self, client):
        resp = client.post("/v1/chat/generation",
                           content_type="application/json", data="")
        assert resp.status_code == 400
