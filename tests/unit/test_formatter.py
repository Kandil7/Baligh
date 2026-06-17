"""Tests for prompt formatting."""

import pytest
from unittest.mock import MagicMock, patch
from baligh.data.formatter import PromptFormatter, get_cpt_formatter, get_sft_formatter


class TestPromptFormatter:
    """Tests for PromptFormatter."""

    @patch('baligh.data.formatter.get_tokenizer')
    def test_init(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        formatter = PromptFormatter(max_seq_length=1024)
        assert formatter.max_seq_length == 1024
        assert formatter.packing is False

    @patch('baligh.data.formatter.get_tokenizer')
    def test_format_cpt(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
        mock_get_tokenizer.return_value = mock_tokenizer
        
        formatter = PromptFormatter()
        result = formatter.format_cpt({"text": "نص تجريبي"})
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["labels"] == result["input_ids"]

    @patch('baligh.data.formatter.get_tokenizer')
    def test_format_cpt_empty_text(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        formatter = PromptFormatter()
        result = formatter.format_cpt({"text": ""})
        
        assert result["input_ids"] == []
        assert result["attention_mask"] == []
        assert result["labels"] == []

    @patch('baligh.data.formatter.get_tokenizer')
    def test_format_sft(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5, 6, 7],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1]
        }
        mock_get_tokenizer.return_value = mock_tokenizer
        
        with patch('baligh.data.formatter.apply_chat_template') as mock_template:
            mock_template.return_value = [1, 2, 3, 4, 5, 6, 7]
            
            formatter = PromptFormatter()
            result = formatter.format_sft({
                "instruction": "اكتب قصيدة",
                "input": "",
                "output": "في الريف حياة"
            })
            
            assert "input_ids" in result
            assert "labels" in result

    @patch('baligh.data.formatter.get_tokenizer')
    def test_call_dispatches_cpt(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        mock_get_tokenizer.return_value = mock_tokenizer
        
        formatter = PromptFormatter()
        batch = {"text": ["نص 1", "نص 2"]}
        result = formatter(batch)
        
        assert "input_ids" in result
        assert len(result["input_ids"]) == 2

    @patch('baligh.data.formatter.get_tokenizer')
    def test_call_dispatches_sft(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        with patch('baligh.data.formatter.apply_chat_template') as mock_template:
            mock_template.return_value = [1, 2, 3, 4, 5]
            
            formatter = PromptFormatter()
            batch = {
                "instruction": ["سؤال 1", "سؤال 2"],
                "input": ["", ""],
                "output": ["جواب 1", "جواب 2"]
            }
            result = formatter(batch)
            
            assert "input_ids" in result


class TestFactoryFunctions:
    """Tests for formatter factory functions."""

    @patch('baligh.data.formatter.get_tokenizer')
    def test_get_cpt_formatter(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = MagicMock()
        
        formatter = get_cpt_formatter()
        assert formatter.packing is True

    @patch('baligh.data.formatter.get_tokenizer')
    def test_get_sft_formatter(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = MagicMock()
        
        formatter = get_sft_formatter()
        assert formatter.packing is False
