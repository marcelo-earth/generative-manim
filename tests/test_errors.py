"""Unit tests for api/errors.py response helpers."""

import pytest


class TestErrorResponse:
    def test_returns_json_with_error_field(self, client):
        from api.errors import error_response
        with client.application.test_request_context():
            resp, status = error_response("something broke", 500)
        assert status == 500
        assert resp.get_json()["error"] == "something broke"

    def test_includes_code_when_provided(self, client):
        from api.errors import error_response
        with client.application.test_request_context():
            resp, status = error_response("oops", 400, code="bad_thing")
        data = resp.get_json()
        assert data["code"] == "bad_thing"
        assert data["error"] == "oops"

    def test_omits_code_when_none(self, client):
        from api.errors import error_response
        with client.application.test_request_context():
            resp, _ = error_response("oops", 400)
        assert "code" not in resp.get_json()


class TestBadRequest:
    def test_status_400(self, client):
        from api.errors import bad_request
        with client.application.test_request_context():
            _, status = bad_request("bad input")
        assert status == 400

    def test_message_in_body(self, client):
        from api.errors import bad_request
        with client.application.test_request_context():
            resp, _ = bad_request("bad input")
        assert resp.get_json()["error"] == "bad input"

    def test_optional_code(self, client):
        from api.errors import bad_request
        with client.application.test_request_context():
            resp, _ = bad_request("bad input", code="invalid_field")
        assert resp.get_json()["code"] == "invalid_field"


class TestUnauthorized:
    def test_status_401(self, client):
        from api.errors import unauthorized
        with client.application.test_request_context():
            _, status = unauthorized()
        assert status == 401

    def test_default_message(self, client):
        from api.errors import unauthorized
        with client.application.test_request_context():
            resp, _ = unauthorized()
        assert "authentication" in resp.get_json()["error"].lower()

    def test_custom_message(self, client):
        from api.errors import unauthorized
        with client.application.test_request_context():
            resp, _ = unauthorized("token expired")
        assert resp.get_json()["error"] == "token expired"


class TestNotFound:
    def test_status_404(self, client):
        from api.errors import not_found
        with client.application.test_request_context():
            _, status = not_found("thing not found")
        assert status == 404

    def test_message_in_body(self, client):
        from api.errors import not_found
        with client.application.test_request_context():
            resp, _ = not_found("model xyz missing")
        assert resp.get_json()["error"] == "model xyz missing"


class TestRateLimited:
    def test_status_429(self, client):
        from api.errors import rate_limited
        with client.application.test_request_context():
            _, status = rate_limited()
        assert status == 429

    def test_default_message(self, client):
        from api.errors import rate_limited
        with client.application.test_request_context():
            resp, _ = rate_limited()
        assert "rate" in resp.get_json()["error"].lower()


class TestGatewayTimeout:
    def test_status_504(self, client):
        from api.errors import gateway_timeout
        with client.application.test_request_context():
            _, status = gateway_timeout()
        assert status == 504

    def test_default_message(self, client):
        from api.errors import gateway_timeout
        with client.application.test_request_context():
            resp, _ = gateway_timeout()
        assert "timed out" in resp.get_json()["error"].lower()

    def test_custom_message(self, client):
        from api.errors import gateway_timeout
        with client.application.test_request_context():
            resp, _ = gateway_timeout("render timed out")
        assert resp.get_json()["error"] == "render timed out"


class TestInternalError:
    def test_status_500(self, client):
        from api.errors import internal_error
        with client.application.test_request_context():
            _, status = internal_error()
        assert status == 500

    def test_default_message(self, client):
        from api.errors import internal_error
        with client.application.test_request_context():
            resp, _ = internal_error()
        assert "error" in resp.get_json()

    def test_custom_message_and_code(self, client):
        from api.errors import internal_error
        with client.application.test_request_context():
            resp, _ = internal_error("db exploded", code="db_error")
        data = resp.get_json()
        assert data["error"] == "db exploded"
        assert data["code"] == "db_error"
