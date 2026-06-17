"""Tests for tokenizer utilities."""

import pytest
from unittest.mock import MagicMock, patch
from baligh.models.tokenizer import (
    get_tokenizer,
    get_chat_template,
    apply_chat_template,
    format_instruction,
    count_tokens,
)


class TestGetTokenizer:
    """Tests for get_tokenizer."""

    @patch('baligh.models.tokenizer.AutoTokenizer')
    def test_default_params(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = get_tokenizer()
        
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        assert tokenizer.padding_side == "right"
        assert tokenizer.truncation_side == "right"

    @patch('baligh.models.tokenizer.AutoTokenizer')
    def test_custom_params(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "eos"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = get_tokenizer(
            tokenizer_name="custom/model",
            max_seq_length=4096,
            padding_side="left"
        )
        
        assert tokenizer.padding_side == "left"
        assert tokenizer.model_max_length == 4096


class TestGetChatTemplate:
    """Tests for get_chat_template."""

    def test_returns_string(self):
        mock_tokenizer = MagicMock()
        template = get_chat_template(mock_tokenizer)
        assert isinstance(template, str)

    def test_contains_roles(self):
        mock_tokenizer = MagicMock()
        template = get_chat_template(mock_tokenizer)
        assert "system" in template
        assert "user" in template
        assert "assistant" in template


class TestFormatInstruction:
    """Tests for format_instruction."""

    def test_instruction_only(self):
        messages = format_instruction("اكتب قصيدة")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "اكتب قصيدة"

    def test_instruction_with_input(self):
        messages = format_instruction("ترجم", "Hello")
        assert len(messages) == 1
        assert "ترجم" in messages[0]["content"]
        assert "Hello" in messages[0]["content"]

    def test_instruction_with_output(self):
        messages = format_instruction("اكتب", output="جواب")
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "جواب"


class TestCountTokens:
    """Tests for count_tokens."""

    @patch('baligh.models.tokenizer.get_tokenizer')
    def test_count(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_get_tokenizer.return_value = mock_tokenizer
        
        count = count_tokens(mock_tokenizer, "Hello world")
        assert count == 5
