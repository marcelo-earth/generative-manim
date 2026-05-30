"""Unit tests for count_images_in_conversation and manage_conversation_images."""

import pytest
from api.routes.chat_generation import count_images_in_conversation, manage_conversation_images


def _text_msg(role, text):
    return {"role": role, "content": text}


def _image_msg(n=1):
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}} for _ in range(n)]
    return {"role": "user", "content": content}


def _mixed_msg(text, n_images=1):
    content = [{"type": "text", "text": text}]
    for _ in range(n_images):
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}})
    return {"role": "user", "content": content}


class TestCountImagesInConversation:
    def test_empty_conversation(self):
        total, indices = count_images_in_conversation([])
        assert total == 0
        assert indices == []

    def test_no_images(self):
        msgs = [_text_msg("user", "hello"), _text_msg("assistant", "hi")]
        total, indices = count_images_in_conversation(msgs)
        assert total == 0
        assert indices == []

    def test_single_image_message(self):
        msgs = [_image_msg(1)]
        total, indices = count_images_in_conversation(msgs)
        assert total == 1
        assert indices == [0]

    def test_multiple_images_in_one_message(self):
        msgs = [_image_msg(3)]
        total, indices = count_images_in_conversation(msgs)
        assert total == 3
        assert indices == [0]

    def test_images_across_multiple_messages(self):
        msgs = [_image_msg(2), _text_msg("assistant", "ok"), _image_msg(1)]
        total, indices = count_images_in_conversation(msgs)
        assert total == 3
        assert indices == [0, 2]

    def test_assistant_image_not_counted(self):
        msg = {"role": "assistant", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]}
        total, indices = count_images_in_conversation([msg])
        assert total == 0

    def test_mixed_content_message(self):
        msgs = [_mixed_msg("describe", n_images=2)]
        total, indices = count_images_in_conversation(msgs)
        assert total == 2
        assert indices == [0]


class TestManageConversationImages:
    def test_non_openai_engine_returns_count_unchanged(self):
        msgs = [_image_msg(5)]
        result = manage_conversation_images(msgs, 10, "anthropic")
        assert result == 10

    def test_non_openai_engine_does_not_remove_messages(self):
        msgs = [_image_msg(5)]
        original_len = len(msgs)
        manage_conversation_images(msgs, 100, "gemini")
        assert len(msgs) == original_len

    def test_openai_within_limit_returns_count(self):
        msgs = [_image_msg(5)]
        result = manage_conversation_images(msgs, 10, "openai")
        assert result == min(50 - 5, 10)

    def test_openai_at_limit_triggers_eviction(self):
        msgs = [_image_msg(40)]
        result = manage_conversation_images(msgs, 15, "openai")
        assert result <= 50

    def test_openai_over_limit_evicts_oldest(self):
        msgs = [_image_msg(30), _image_msg(25)]
        original_len = len(msgs)
        manage_conversation_images(msgs, 5, "openai")
        assert len(msgs) < original_len

    def test_openai_no_existing_images_returns_new_count(self):
        msgs = [_text_msg("user", "hello")]
        result = manage_conversation_images(msgs, 20, "openai")
        assert result == 20

    def test_featherless_returns_count_unchanged(self):
        msgs = []
        result = manage_conversation_images(msgs, 7, "featherless")
        assert result == 7

    def test_litellm_returns_count_unchanged(self):
        result = manage_conversation_images([], 3, "litellm")
        assert result == 3
