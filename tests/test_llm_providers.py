"""Tests for pure utility functions in api/llm_providers.py."""

import sys
import types
from unittest import mock

import pytest

# Stub google.genai before any app imports
_fake_genai = types.ModuleType("google.genai")
_fake_google = types.ModuleType("google")
_fake_google.genai = _fake_genai
sys.modules.setdefault("google", _fake_google)
sys.modules.setdefault("google.genai", _fake_genai)

from api.llm_providers import (
    normalize_gemini_model,
    strip_markdown_code_fence,
    generate_gemini_content,
    GEMINI_MODEL_ALIASES,
)


class TestNormalizeGeminiModel:
    def test_known_alias_flash(self):
        assert normalize_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash"

    def test_known_alias_gemini3_short(self):
        assert normalize_gemini_model("gemini-3-flash") == "gemini-3-flash-preview"

    def test_known_alias_gemini3_full(self):
        assert normalize_gemini_model("gemini-3-flash-preview") == "gemini-3-flash-preview"

    def test_unknown_model_passed_through(self):
        assert normalize_gemini_model("gemini-future-model") == "gemini-future-model"

    def test_empty_string_passed_through(self):
        assert normalize_gemini_model("") == ""

    def test_all_aliases_resolve(self):
        for alias, expected in GEMINI_MODEL_ALIASES.items():
            assert normalize_gemini_model(alias) == expected


class TestStripMarkdownCodeFence:
    def test_no_fence_returned_as_is(self):
        code = "from manim import *\nclass GenScene(Scene): pass"
        assert strip_markdown_code_fence(code) == code

    def test_generic_fence_stripped(self):
        fenced = "```\nfrom manim import *\n```"
        assert strip_markdown_code_fence(fenced) == "from manim import *"

    def test_language_tagged_fence_stripped(self):
        fenced = "```python\nfrom manim import *\n```"
        assert strip_markdown_code_fence(fenced) == "from manim import *"

    def test_leading_trailing_whitespace_stripped(self):
        fenced = "  ```python\ncode\n```  "
        assert strip_markdown_code_fence(fenced) == "code"

    def test_multiline_code_preserved(self):
        fenced = "```python\nline1\nline2\nline3\n```"
        result = strip_markdown_code_fence(fenced)
        assert result == "line1\nline2\nline3"

    def test_empty_string_returns_empty(self):
        assert strip_markdown_code_fence("") == ""

    def test_none_returns_empty(self):
        assert strip_markdown_code_fence(None) == ""

    def test_fence_without_closing_backticks(self):
        fenced = "```python\ncode line"
        result = strip_markdown_code_fence(fenced)
        assert result == "code line"

    def test_only_backtick_lines_returns_empty(self):
        fenced = "```\n```"
        assert strip_markdown_code_fence(fenced) == ""


class TestGenerateGeminiContent:
    def test_raises_when_genai_unavailable(self, monkeypatch):
        import api.llm_providers as providers
        monkeypatch.setattr(providers, "genai", None)
        with pytest.raises(RuntimeError, match="google-genai is required"):
            generate_gemini_content("gemini-2.5-flash", "system", "prompt")

    def test_raises_when_api_key_missing(self, monkeypatch):
        import api.llm_providers as providers
        monkeypatch.setattr(providers, "genai", mock.MagicMock())
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            generate_gemini_content("gemini-2.5-flash", "system", "prompt")

    def test_calls_generate_content_with_combined_prompt(self, monkeypatch):
        fake_genai = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.text = "from manim import *"
        fake_genai.Client.return_value.models.generate_content.return_value = fake_response

        import api.llm_providers as providers
        monkeypatch.setattr(providers, "genai", fake_genai)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        result = generate_gemini_content("gemini-2.5-flash", "be helpful", "draw a circle")

        fake_genai.Client.assert_called_once_with(api_key="test-key")
        call_kwargs = fake_genai.Client.return_value.models.generate_content.call_args[1]
        assert "be helpful" in call_kwargs["contents"]
        assert "draw a circle" in call_kwargs["contents"]
        assert result == "from manim import *"

    def test_model_alias_applied(self, monkeypatch):
        fake_genai = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.text = "code"
        fake_genai.Client.return_value.models.generate_content.return_value = fake_response

        import api.llm_providers as providers
        monkeypatch.setattr(providers, "genai", fake_genai)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        generate_gemini_content("gemini-3-flash", "sys", "prompt")

        call_kwargs = fake_genai.Client.return_value.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-3-flash-preview"

    def test_code_fence_stripped_from_response(self, monkeypatch):
        fake_genai = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.text = "```python\nfrom manim import *\n```"
        fake_genai.Client.return_value.models.generate_content.return_value = fake_response

        import api.llm_providers as providers
        monkeypatch.setattr(providers, "genai", fake_genai)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        result = generate_gemini_content("gemini-2.5-flash", "sys", "prompt")
        assert result == "from manim import *"
