"""Request body validation helpers for the Flask API routes."""

from flask import jsonify, request


VALID_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}


def get_json_body():
    """Return parsed JSON body or a 400 error response tuple.

    Callers should check: ``body, err = get_json_body(); if err: return err``
    """
    body = request.get_json(silent=True)
    if body is None:
        return None, (jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400)
    if not isinstance(body, dict):
        return None, (jsonify({"error": "Request body must be a JSON object"}), 400)
    return body, None


def require_string(body, field, allow_empty=False):
    """Validate that *field* exists and is a non-empty string.

    Returns ``(value, None)`` on success or ``(None, error_response)`` on failure.
    """
    value = body.get(field)
    if value is None:
        return None, (jsonify({"error": f"'{field}' is required"}), 400)
    if not isinstance(value, str):
        return None, (jsonify({"error": f"'{field}' must be a string"}), 400)
    if not allow_empty and not value.strip():
        return None, (jsonify({"error": f"'{field}' must not be empty"}), 400)
    return value, None


def validate_aspect_ratio(body, default="16:9"):
    """Validate the optional 'aspect_ratio' field.

    Returns ``(value, None)`` on success or ``(default, error_response)`` if invalid.
    """
    value = body.get("aspect_ratio", default)
    if value not in VALID_ASPECT_RATIOS:
        return None, (
            jsonify({"error": f"'aspect_ratio' must be one of: {', '.join(sorted(VALID_ASPECT_RATIOS))}"}),
            400,
        )
    return value, None


def validate_boolean(body, field, default=False):
    """Validate an optional boolean field.

    Returns ``(value, None)`` on success or ``(None, error_response)`` if invalid.
    """
    value = body.get(field, default)
    if not isinstance(value, bool):
        return None, (jsonify({"error": f"'{field}' must be a boolean"}), 400)
    return value, None
