"""Shared pytest fixtures and stubs for the generative-manim test suite."""

import sys
import types
from unittest import mock

import pytest


def _install_litellm_stub():
    """Register a fake litellm module so app imports don't require the real package."""
    if "litellm" in sys.modules:
        return

    fake_litellm = types.ModuleType("litellm")
    fake_exceptions = types.ModuleType("litellm.exceptions")

    class _AuthenticationError(Exception):
        pass

    class _NotFoundError(Exception):
        pass

    class _RateLimitError(Exception):
        pass

    class _Timeout(Exception):
        pass

    fake_exceptions.AuthenticationError = _AuthenticationError
    fake_exceptions.NotFoundError = _NotFoundError
    fake_exceptions.RateLimitError = _RateLimitError
    fake_exceptions.Timeout = _Timeout
    fake_litellm.exceptions = fake_exceptions
    fake_litellm.completion = mock.MagicMock()

    sys.modules["litellm"] = fake_litellm
    sys.modules["litellm.exceptions"] = fake_exceptions


_install_litellm_stub()


@pytest.fixture
def app():
    sys.path.insert(0, ".")
    from api.run import app as flask_app
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()
