"""Tests for structured output extraction."""

import json
from unittest.mock import MagicMock, patch

from baligh.inference.structured import StructuredOutput


class TestStructuredOutput:
    def test_extract_json_valid(self):
        with patch("baligh.inference.structured.TextGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = '{"key": "value"}'
            MockGen.return_value = mock_gen

            so = StructuredOutput("fake_model")
            result = so.extract_json("test prompt")
            assert result == {"key": "value"}

    def test_extract_json_in_text(self):
        with patch("baligh.inference.structured.TextGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = '{"key": "value"}'
            MockGen.return_value = mock_gen

            so = StructuredOutput("fake_model")
            result = so.extract_json("test prompt")
            assert result == {"key": "value"}

    def test_extract_json_array(self):
        with patch("baligh.inference.structured.TextGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = '[1, 2, 3]'
            MockGen.return_value = mock_gen

            so = StructuredOutput("fake_model")
            result = so.extract_json("test prompt")
            assert result == [1, 2, 3]

    def test_extract_json_nested(self):
        with patch("baligh.inference.structured.TextGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = '{"a": {"b": 1}}'
            MockGen.return_value = mock_gen

            so = StructuredOutput("fake_model")
            result = so.extract_json("test prompt")
            assert result == {"a": {"b": 1}}

    def test_extract_json_retries(self):
        with patch("baligh.inference.structured.TextGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate.side_effect = ["invalid", "also invalid", '{"ok": true}']
            MockGen.return_value = mock_gen

            so = StructuredOutput("fake_model")
            result = so.extract_json("test prompt", max_retries=3)
            assert result == {"ok": True}
            assert mock_gen.generate.call_count == 3

    def test_extract_json_fails_after_retries(self):
        with patch("baligh.inference.structured.TextGenerator") as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = "always invalid"
            MockGen.return_value = mock_gen

            so = StructuredOutput("fake_model")
            try:
                so.extract_json("test prompt", max_retries=2)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Failed" in str(e)
