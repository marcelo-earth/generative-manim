"""Simple API key authentication for the Flask API."""

import hmac
import os

from flask import jsonify, request

_EXEMPT_PATHS = {"/", "/openapi.yaml", "/health", "/v1/health"}


def require_api_key():
    """Flask before_request hook enforcing an API key on protected routes.

    Reads API_KEY from the environment on every call (not at import time)
    so tests and tools can toggle it without reloading the app. If API_KEY
    is unset, authentication is not enforced, which keeps local development
    working without extra setup. Set API_KEY in any deployment reachable
    from the public internet: unauthenticated access lets any caller run
    arbitrary code through the rendering endpoints.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        return None

    if request.path in _EXEMPT_PATHS or request.path.startswith("/public/"):
        return None

    provided = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(provided, api_key):
        return jsonify({"error": "Invalid or missing API key"}), 401

    return None
