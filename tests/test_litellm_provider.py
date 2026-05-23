"""Tests for LiteLLM engine integration in generative-manim API."""

import json
import sys
import types
from unittest import mock

import pytest

# Stub litellm before any app imports
_fake_litellm = types.ModuleType("litellm")
_fake_exceptions = types.ModuleType("litellm.exceptions")


class _AuthenticationError(Exception):
    pass


class _NotFoundError(Exception):
    pass


class _RateLimitError(Exception):
    pass


class _Timeout(Exception):
    pass


_fake_exceptions.AuthenticationError = _AuthenticationError
_fake_exceptions.NotFoundError = _NotFoundError
_fake_exceptions.RateLimitError = _RateLimitError
_fake_exceptions.Timeout = _Timeout

_fake_litellm.exceptions = _fake_exceptions
_fake_litellm.completion = mock.MagicMock()

sys.modules["litellm"] = _fake_litellm
sys.modules["litellm.exceptions"] = _fake_exceptions


def _make_response(content):
    msg = mock.MagicMock()
    msg.content = content
    choice = mock.MagicMock()
    choice.message = msg
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


def _make_streaming_response(text):
    chunks = []
    for char in text:
        chunk = mock.MagicMock()
        chunk.choices = [mock.MagicMock()]
        chunk.choices[0].delta = mock.MagicMock()
        chunk.choices[0].delta.content = char
        chunks.append(chunk)
    return iter(chunks)


@pytest.fixture(autouse=True)
def reset_mocks():
    _fake_litellm.completion.reset_mock()
    _fake_litellm.completion.side_effect = None
    _fake_litellm.completion.return_value = _make_response("from manim import *\nclass GenScene(Scene):\n  pass")
    yield


@pytest.fixture
def app():
    sys.path.insert(0, ".")
    from api.run import app
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestCodeGenerationLiteLLM:
    """Tests for the /v1/code/generation endpoint with litellm engine."""

    def test_basic_code_generation(self, client):
        resp = client.post("/v1/code/generation", json={
            "prompt": "Create a blue circle",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "code" in data
        _fake_litellm.completion.assert_called_once()

    def test_drop_params_true(self, client):
        client.post("/v1/code/generation", json={
            "prompt": "Create a circle",
            "engine": "litellm",
            "model": "anthropic/claude-sonnet-4-6",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["drop_params"] is True

    def test_model_forwarded(self, client):
        client.post("/v1/code/generation", json={
            "prompt": "Create a circle",
            "engine": "litellm",
            "model": "groq/llama-3.3-70b-versatile",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "groq/llama-3.3-70b-versatile"

    def test_provider_prefixed_model(self, client):
        client.post("/v1/code/generation", json={
            "prompt": "Create a circle",
            "engine": "litellm",
            "model": "anthropic/claude-haiku-4-5",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert "/" in call_kwargs["model"]

    def test_default_model_when_not_specified(self, client):
        client.post("/v1/code/generation", json={
            "prompt": "Create a circle",
            "engine": "litellm",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o"

    def test_api_key_from_env(self, client, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-test-key")
        client.post("/v1/code/generation", json={
            "prompt": "Create a circle",
            "engine": "litellm",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key"

    def test_api_key_omitted_when_not_set(self, client, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        client.post("/v1/code/generation", json={
            "prompt": "Create a circle",
            "engine": "litellm",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert "api_key" not in call_kwargs

    def test_messages_structure(self, client):
        client.post("/v1/code/generation", json={
            "prompt": "Create a blue circle",
            "engine": "litellm",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "Manim" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "blue circle" in messages[1]["content"]


class TestCodeGenerationLiteLLMErrors:
    """Tests for litellm-specific error handling."""

    def test_auth_error_returns_401(self, client):
        _fake_litellm.completion.side_effect = _AuthenticationError("Invalid key")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 401
        assert "auth" in resp.get_json()["error"].lower()

    def test_not_found_returns_404(self, client):
        _fake_litellm.completion.side_effect = _NotFoundError("Model not found")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 404

    def test_rate_limit_returns_429(self, client):
        _fake_litellm.completion.side_effect = _RateLimitError("429")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 429

    def test_timeout_returns_504(self, client):
        _fake_litellm.completion.side_effect = _Timeout("timed out")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 504

    def test_generic_error_returns_500(self, client):
        _fake_litellm.completion.side_effect = RuntimeError("unexpected")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 500


class TestCodeGenerationLiteLLMEdgeCases:
    """Edge case tests for empty/null responses, streaming, etc."""

    def test_empty_response_content(self, client):
        _fake_litellm.completion.return_value = _make_response("")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 200
        assert resp.get_json()["code"] == ""

    def test_null_response_content(self, client):
        _fake_litellm.completion.return_value = _make_response(None)
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 200

    def test_no_choices_raises_500(self, client):
        bad_resp = mock.MagicMock()
        bad_resp.choices = []
        _fake_litellm.completion.return_value = bad_resp
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 500

    def test_context_length_exceeded(self, client):
        """Token limit / context window overflow returns 500."""
        _fake_litellm.completion.side_effect = RuntimeError("context_length_exceeded")
        resp = client.post("/v1/code/generation", json={
            "prompt": "test",
            "engine": "litellm",
        })
        assert resp.status_code == 500
        assert "context_length" in resp.get_json()["error"]


class TestChatGenerationLiteLLMEngine:
    """Tests for litellm streaming in chat generation."""

    def test_litellm_engine_accepted(self, client):
        _fake_litellm.completion.return_value = _make_streaming_response("Hello!")
        resp = client.post("/v1/chat/generation", json={
            "prompt": "Create a circle animation",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        assert resp.status_code == 200

    def test_litellm_invalid_engine_rejected(self, client):
        resp = client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "nonexistent_engine",
        })
        assert resp.status_code == 400

    def test_streaming_response_content(self, client):
        _fake_litellm.completion.return_value = _make_streaming_response("Hello world")
        resp = client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        assert resp.status_code == 200
        assert "Hello world" in resp.get_data(as_text=True)

    def test_streaming_partial_chunks(self, client):
        """Verify None content chunks are skipped gracefully."""
        chunks = []
        for text in ["Hello", None, " world", None]:
            chunk = mock.MagicMock()
            chunk.choices = [mock.MagicMock()]
            chunk.choices[0].delta = mock.MagicMock()
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        _fake_litellm.completion.return_value = iter(chunks)

        resp = client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "Hello" in body
        assert "world" in body

    def test_streaming_empty_response(self, client):
        """Empty stream returns 200 with no content."""
        final = mock.MagicMock()
        final.choices = [mock.MagicMock()]
        final.choices[0].delta = mock.MagicMock()
        final.choices[0].delta.content = None
        _fake_litellm.completion.return_value = iter([final])

        resp = client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        assert resp.status_code == 200

    def test_streaming_error_handled(self, client):
        """Error during streaming returns error message."""
        _fake_litellm.completion.side_effect = RuntimeError("connection lost")
        resp = client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        assert resp.status_code == 200
        assert "Error" in resp.get_data(as_text=True)

    def test_streaming_drop_params(self, client):
        _fake_litellm.completion.return_value = _make_streaming_response("ok")
        client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "litellm",
            "model": "openai/gpt-4o",
        })
        call_kwargs = _fake_litellm.completion.call_args[1]
        assert call_kwargs["drop_params"] is True

    def test_platform_mode_json_format(self, client):
        _fake_litellm.completion.return_value = _make_streaming_response("Hi")
        resp = client.post("/v1/chat/generation", json={
            "prompt": "test",
            "engine": "litellm",
            "model": "openai/gpt-4o",
            "isForPlatform": True,
        })
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert '"type": "text"' in body
