"""Unit tests for api/video_utils.py."""

from api.video_utils import get_frame_config


class TestGetFrameConfig:
    def test_16x9_returns_4k_resolution(self):
        size, width = get_frame_config("16:9")
        assert size == (3840, 2160)
        assert width == 14.22

    def test_9x16_returns_portrait_resolution(self):
        size, width = get_frame_config("9:16")
        assert size == (1080, 1920)
        assert width == 8.0

    def test_1x1_returns_square_resolution(self):
        size, width = get_frame_config("1:1")
        assert size == (1080, 1080)
        assert width == 8.0

    def test_unknown_ratio_falls_back_to_16x9(self):
        size, width = get_frame_config("4:3")
        assert size == (3840, 2160)
        assert width == 14.22

    def test_empty_string_falls_back_to_16x9(self):
        size, width = get_frame_config("")
        assert size == (3840, 2160)
