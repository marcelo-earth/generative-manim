"""Tests for the /v1/models endpoint."""

import sys
import types
from unittest import mock

import pytest

_fake_litellm = types.ModuleType("litellm")
_fake_exceptions = types.ModuleType("litellm.exceptions")
_fake_exceptions.AuthenticationError = Exception
_fake_exceptions.NotFoundError = Exception
_fake_exceptions.RateLimitError = Exception
_fake_exceptions.Timeout = Exception
_fake_litellm.exceptions = _fake_exceptions
_fake_litellm.completion = mock.MagicMock()
sys.modules.setdefault("litellm", _fake_litellm)
sys.modules.setdefault("litellm.exceptions", _fake_exceptions)


@pytest.fixture
def app():
    sys.path.insert(0, ".")
    from api.run import app
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestModelsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_response_has_engines_key(self, client):
        data = client.get("/v1/models").get_json()
        assert "engines" in data
        assert isinstance(data["engines"], list)

    def test_all_five_engines_present(self, client):
        data = client.get("/v1/models").get_json()
        engine_names = {e["engine"] for e in data["engines"]}
        assert engine_names == {"openai", "anthropic", "gemini", "featherless", "litellm"}

    def test_engine_shape(self, client):
        data = client.get("/v1/models").get_json()
        for entry in data["engines"]:
            assert "engine" in entry
            assert "configured" in entry
            assert "default" in entry
            assert "models" in entry
            assert isinstance(entry["configured"], bool)
            assert isinstance(entry["models"], list)

    def test_configured_false_when_no_key(self, client, monkeypatch):
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                    "FEATHERLESS_API_KEY", "LITELLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        data = client.get("/v1/models").get_json()
        for entry in data["engines"]:
            assert entry["configured"] is False, f"{entry['engine']} should not be configured"

    def test_configured_true_when_key_set(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        data = client.get("/v1/models").get_json()
        openai_entry = next(e for e in data["engines"] if e["engine"] == "openai")
        assert openai_entry["configured"] is True

    def test_anthropic_default_is_claude_sonnet_4_6(self, client):
        data = client.get("/v1/models").get_json()
        anthropic = next(e for e in data["engines"] if e["engine"] == "anthropic")
        assert anthropic["default"] == "claude-sonnet-4-6"

    def test_anthropic_models_include_claude_4(self, client):
        data = client.get("/v1/models").get_json()
        anthropic = next(e for e in data["engines"] if e["engine"] == "anthropic")
        model_ids = [m["id"] for m in anthropic["models"]]
        assert "claude-sonnet-4-6" in model_ids
        assert "claude-opus-4-7" in model_ids

    def test_litellm_has_note(self, client):
        data = client.get("/v1/models").get_json()
        litellm = next(e for e in data["engines"] if e["engine"] == "litellm")
        assert "note" in litellm
