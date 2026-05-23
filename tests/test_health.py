"""Tests for the /health and /v1/health endpoints."""

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


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_v1_health_returns_200(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_health_response_has_status_field(self, client):
        data = client.get("/health").get_json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_response_has_providers(self, client):
        data = client.get("/health").get_json()
        assert "providers" in data
        providers = data["providers"]
        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" in providers

    def test_health_provider_shape(self, client):
        data = client.get("/health").get_json()
        for name, info in data["providers"].items():
            assert "configured" in info, f"provider {name} missing 'configured'"
            assert "env_var" in info, f"provider {name} missing 'env_var'"
            assert isinstance(info["configured"], bool)

    def test_healthy_when_key_configured(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        data = client.get("/health").get_json()
        assert data["status"] == "healthy"
        assert data["providers"]["openai"]["configured"] is True

    def test_degraded_when_no_keys_configured(self, client, monkeypatch):
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                    "FEATHERLESS_API_KEY", "LITELLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        data = client.get("/health").get_json()
        assert data["status"] == "degraded"
        assert data["configured_providers"] == 0

    def test_configured_providers_count(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k2")
        for var in ("GEMINI_API_KEY", "FEATHERLESS_API_KEY", "LITELLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        data = client.get("/health").get_json()
        assert data["configured_providers"] == 2
        assert data["total_providers"] == 5
