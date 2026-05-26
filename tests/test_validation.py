"""Unit tests for api/validation.py helpers."""

import pytest


class TestGetJsonBody:
    def test_valid_json_returns_dict(self, client):
        with client.application.test_request_context(
            "/", method="POST", json={"key": "value"}
        ):
            from api.validation import get_json_body
            body, err = get_json_body()
        assert err is None
        assert body == {"key": "value"}

    def test_missing_body_returns_error(self, client):
        with client.application.test_request_context(
            "/", method="POST", content_type="application/json", data=""
        ):
            from api.validation import get_json_body
            body, err = get_json_body()
        assert body is None
        assert err is not None

    def test_non_object_json_returns_error(self, client):
        with client.application.test_request_context(
            "/", method="POST", content_type="application/json", data="[1, 2, 3]"
        ):
            from api.validation import get_json_body
            body, err = get_json_body()
        assert body is None
        assert err is not None

    def test_wrong_content_type_returns_error(self, client):
        with client.application.test_request_context(
            "/", method="POST", content_type="text/plain", data="hello"
        ):
            from api.validation import get_json_body
            body, err = get_json_body()
        assert body is None
        assert err is not None


class TestRequireString:
    def test_valid_string_returned(self, client):
        from api.validation import require_string
        with client.application.test_request_context("/"):
            value, err = require_string({"name": "hello"}, "name")
        assert err is None
        assert value == "hello"

    def test_missing_field_returns_error(self, client):
        from api.validation import require_string
        with client.application.test_request_context("/"):
            value, err = require_string({}, "name")
        assert value is None
        assert err is not None

    def test_non_string_field_returns_error(self, client):
        from api.validation import require_string
        with client.application.test_request_context("/"):
            value, err = require_string({"name": 42}, "name")
        assert value is None
        assert err is not None

    def test_empty_string_returns_error_by_default(self, client):
        from api.validation import require_string
        with client.application.test_request_context("/"):
            value, err = require_string({"name": "   "}, "name")
        assert value is None
        assert err is not None

    def test_empty_string_allowed_with_flag(self, client):
        from api.validation import require_string
        with client.application.test_request_context("/"):
            value, err = require_string({"name": ""}, "name", allow_empty=True)
        assert err is None
        assert value == ""

    def test_error_references_field_name(self, client):
        from api.validation import require_string
        with client.application.test_request_context("/"):
            _, (resp, _status) = require_string({}, "prompt")
        assert "prompt" in resp.get_json()["error"]


class TestValidateAspectRatio:
    def test_valid_16x9(self, client):
        from api.validation import validate_aspect_ratio
        with client.application.test_request_context("/"):
            value, err = validate_aspect_ratio({"aspect_ratio": "16:9"})
        assert err is None
        assert value == "16:9"

    def test_valid_9x16(self, client):
        from api.validation import validate_aspect_ratio
        with client.application.test_request_context("/"):
            value, err = validate_aspect_ratio({"aspect_ratio": "9:16"})
        assert err is None
        assert value == "9:16"

    def test_valid_1x1(self, client):
        from api.validation import validate_aspect_ratio
        with client.application.test_request_context("/"):
            value, err = validate_aspect_ratio({"aspect_ratio": "1:1"})
        assert err is None
        assert value == "1:1"

    def test_missing_uses_default(self, client):
        from api.validation import validate_aspect_ratio
        with client.application.test_request_context("/"):
            value, err = validate_aspect_ratio({})
        assert err is None
        assert value == "16:9"

    def test_invalid_ratio_returns_error(self, client):
        from api.validation import validate_aspect_ratio
        with client.application.test_request_context("/"):
            value, err = validate_aspect_ratio({"aspect_ratio": "4:3"})
        assert value is None
        assert err is not None

    def test_invalid_ratio_error_mentions_field(self, client):
        from api.validation import validate_aspect_ratio
        with client.application.test_request_context("/"):
            _, (resp, _status) = validate_aspect_ratio({"aspect_ratio": "bad"})
        assert "aspect_ratio" in resp.get_json()["error"]


class TestValidateBoolean:
    def test_true_accepted(self, client):
        from api.validation import validate_boolean
        with client.application.test_request_context("/"):
            value, err = validate_boolean({"flag": True}, "flag")
        assert err is None
        assert value is True

    def test_false_accepted(self, client):
        from api.validation import validate_boolean
        with client.application.test_request_context("/"):
            value, err = validate_boolean({"flag": False}, "flag")
        assert err is None
        assert value is False

    def test_missing_returns_default_false(self, client):
        from api.validation import validate_boolean
        with client.application.test_request_context("/"):
            value, err = validate_boolean({}, "flag")
        assert err is None
        assert value is False

    def test_missing_returns_custom_default(self, client):
        from api.validation import validate_boolean
        with client.application.test_request_context("/"):
            value, err = validate_boolean({}, "flag", default=True)
        assert err is None
        assert value is True

    def test_string_returns_error(self, client):
        from api.validation import validate_boolean
        with client.application.test_request_context("/"):
            value, err = validate_boolean({"flag": "true"}, "flag")
        assert value is None
        assert err is not None

    def test_integer_returns_error(self, client):
        from api.validation import validate_boolean
        with client.application.test_request_context("/"):
            value, err = validate_boolean({"flag": 1}, "flag")
        assert value is None
        assert err is not None
