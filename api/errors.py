"""Centralized error response helpers for consistent API error shapes."""

from __future__ import annotations

from flask import jsonify


def error_response(message: str, status: int, code: str | None = None):
    """Return a JSON error response with a consistent shape.

    All API errors have at minimum: {"error": "..."}
    When a machine-readable code is provided: {"error": "...", "code": "..."}
    """
    body = {"error": message}
    if code:
        body["code"] = code
    return jsonify(body), status


def bad_request(message: str, code: str | None = None):
    return error_response(message, 400, code)


def unauthorized(message: str = "Authentication failed", code: str | None = None):
    return error_response(message, 401, code)


def not_found(message: str, code: str | None = None):
    return error_response(message, 404, code)


def rate_limited(message: str = "Rate limit exceeded", code: str | None = None):
    return error_response(message, 429, code)


def gateway_timeout(message: str = "Upstream request timed out", code: str | None = None):
    return error_response(message, 504, code)


def internal_error(message: str = "An unexpected error occurred", code: str | None = None):
    return error_response(message, 500, code)
