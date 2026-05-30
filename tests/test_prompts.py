"""Unit tests for api/prompts/system.py shared prompt content."""

from api.prompts.system import MANIM_CODE_GENERATION_PROMPT


class TestManimCodeGenerationPrompt:
    def test_is_string(self):
        assert isinstance(MANIM_CODE_GENERATION_PROMPT, str)

    def test_not_empty(self):
        assert len(MANIM_CODE_GENERATION_PROMPT) > 0

    def test_mentions_genscene(self):
        assert "GenScene" in MANIM_CODE_GENERATION_PROMPT

    def test_mentions_self_play(self):
        assert "self.play" in MANIM_CODE_GENERATION_PROMPT

    def test_contains_manim_import(self):
        assert "from manim import" in MANIM_CODE_GENERATION_PROMPT

    def test_no_invalid_escape_sequences(self):
        import ast
        try:
            ast.literal_eval(f'"{MANIM_CODE_GENERATION_PROMPT}"')
        except (ValueError, SyntaxError):
            pass

    def test_has_rules_section(self):
        assert "Rules" in MANIM_CODE_GENERATION_PROMPT

    def test_no_emdash(self):
        assert "—" not in MANIM_CODE_GENERATION_PROMPT
