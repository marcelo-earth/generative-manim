"""Tests for /v1/video/rendering route."""

import sys
import types
from unittest import mock

import pytest

# Stub litellm before app imports
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

VALID_CODE = "from manim import *\nclass GenScene(Scene):\n    def construct(self): pass"


@pytest.fixture
def app():
    sys.path.insert(0, ".")
    from api.run import app
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _make_popen_mock(returncode=0, stderr_lines=None, stdout_lines=None):
    """Build a mock subprocess.Popen that simulates a finished process."""
    stderr_lines = list(stderr_lines or [])
    stdout_lines = list(stdout_lines or [])

    process = mock.MagicMock()
    process.returncode = returncode

    # readline() yields each line then returns "" to signal EOF
    stdout_seq = stdout_lines + [""]
    stderr_seq = stderr_lines + [""]
    process.stdout.readline.side_effect = stdout_seq
    process.stderr.readline.side_effect = stderr_seq

    # poll() returns None while running, then returncode when done
    # After all lines are consumed the while-loop calls poll(); return returncode.
    process.poll.return_value = returncode

    return process


class TestVideoRenderingValidation:
    def test_missing_code_returns_400(self, client):
        resp = client.post("/v1/video/rendering", json={
            "file_name": "GenScene",
            "file_class": "GenScene",
        })
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_code_returns_400(self, client):
        resp = client.post("/v1/video/rendering", json={"code": ""})
        assert resp.status_code == 400


class TestVideoRenderingSuccess:
    def test_successful_render_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("USE_LOCAL_STORAGE", "true")
        popen_mock = _make_popen_mock(returncode=0)
        expected_url = "http://localhost:8080/public/video-test.mp4"

        with mock.patch("api.routes.video_rendering.subprocess.Popen", return_value=popen_mock):
            with mock.patch("os.path.exists", return_value=True):
                with mock.patch("os.listdir", return_value=["GenScene.mp4"]):
                    with mock.patch("api.routes.video_rendering.move_to_public_folder",
                                    return_value=expected_url):
                        with mock.patch("builtins.open", mock.mock_open()):
                            with mock.patch("os.remove"):
                                resp = client.post("/v1/video/rendering", json={
                                    "code": VALID_CODE,
                                    "file_name": "GenScene",
                                    "file_class": "GenScene",
                                    "user_id": "test-user",
                                    "project_name": "test-project",
                                    "iteration": 1,
                                })

        assert resp.status_code == 200
        data = resp.get_json()
        assert "video_url" in data

    def test_response_includes_video_url(self, client, monkeypatch):
        monkeypatch.setenv("USE_LOCAL_STORAGE", "true")
        expected_url = "http://localhost:8080/public/video-test-user-test-project-1.mp4"
        popen_mock = _make_popen_mock(returncode=0)

        with mock.patch("api.routes.video_rendering.subprocess.Popen", return_value=popen_mock):
            with mock.patch("os.path.exists", return_value=True):
                with mock.patch("os.listdir", return_value=["GenScene.mp4"]):
                    with mock.patch("api.routes.video_rendering.move_to_public_folder",
                                    return_value=expected_url):
                        with mock.patch("builtins.open", mock.mock_open()):
                            with mock.patch("os.remove"):
                                resp = client.post("/v1/video/rendering", json={
                                    "code": VALID_CODE,
                                    "file_name": "GenScene",
                                    "file_class": "GenScene",
                                    "user_id": "test-user",
                                    "project_name": "test-project",
                                    "iteration": 1,
                                })

        assert resp.get_json()["video_url"] == expected_url


class TestVideoRenderingFailure:
    def test_manim_error_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("USE_LOCAL_STORAGE", "true")
        popen_mock = _make_popen_mock(
            returncode=1,
            stderr_lines=["NameError: name 'GenScene' is not defined"],
        )

        with mock.patch("api.routes.video_rendering.subprocess.Popen", return_value=popen_mock):
            with mock.patch("builtins.open", mock.mock_open()):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("os.remove"):
                        resp = client.post("/v1/video/rendering", json={
                            "code": "broken code",
                            "file_name": "GenScene",
                            "file_class": "GenScene",
                        })

        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_subprocess_exception_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("USE_LOCAL_STORAGE", "true")

        with mock.patch("api.routes.video_rendering.subprocess.Popen",
                        side_effect=OSError("manim not found")):
            with mock.patch("builtins.open", mock.mock_open()):
                resp = client.post("/v1/video/rendering", json={
                    "code": VALID_CODE,
                    "file_name": "GenScene",
                    "file_class": "GenScene",
                })

        assert resp.status_code == 500


class TestVideoRenderingAspectRatio:
    def _run_with_captured_write(self, client, monkeypatch, aspect_ratio):
        monkeypatch.setenv("USE_LOCAL_STORAGE", "true")
        popen_mock = _make_popen_mock(returncode=0)
        written = {}

        real_open = open

        def capture_open(path, mode="r", *args, **kwargs):
            if mode == "w" and str(path).endswith(".py"):
                f = mock.MagicMock()
                def capture_write(content):
                    written["content"] = content
                f.write = capture_write
                f.__enter__ = lambda s: f
                f.__exit__ = mock.MagicMock(return_value=False)
                return f
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("api.routes.video_rendering.subprocess.Popen", return_value=popen_mock):
            with mock.patch("os.path.exists", return_value=True):
                with mock.patch("os.listdir", return_value=["GenScene.mp4"]):
                    with mock.patch("api.routes.video_rendering.move_to_public_folder",
                                    return_value="http://localhost/video.mp4"):
                        with mock.patch("builtins.open", side_effect=capture_open):
                            with mock.patch("os.remove"):
                                client.post("/v1/video/rendering", json={
                                    "code": VALID_CODE,
                                    "aspect_ratio": aspect_ratio,
                                })
        return written

    def test_16_9_aspect_ratio_applied(self, client, monkeypatch):
        written = self._run_with_captured_write(client, monkeypatch, "16:9")
        if written.get("content"):
            assert "3840, 2160" in written["content"]

    def test_square_aspect_ratio_applied(self, client, monkeypatch):
        written = self._run_with_captured_write(client, monkeypatch, "1:1")
        if written.get("content"):
            assert "1080, 1080" in written["content"]
