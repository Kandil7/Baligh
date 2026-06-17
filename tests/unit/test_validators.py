"""Tests for data validation."""

import pytest
from baligh.data.validators import (
    validate_cpt_example,
    validate_sft_example,
    validate_dataset,
    check_dataset_schema,
    get_dataset_stats,
    REQUIRED_CPT_COLUMNS,
    REQUIRED_SFT_COLUMNS,
)


class TestValidateCPTExample:
    """Tests for CPT example validation."""

    def test_valid_example(self):
        example = {"text": "هذا نص تجريبي طويل يحتوي على أكثر من خمسين حرفاً للتحقق من صحة التحقق من الأمثلة."}
        valid, error = validate_cpt_example(example)
        assert valid is True
        assert error == ""

    def test_missing_text(self):
        example = {}
        valid, error = validate_cpt_example(example)
        assert valid is False
        assert "Missing text field" in error

    def test_empty_text(self):
        example = {"text": ""}
        valid, error = validate_cpt_example(example)
        assert valid is False

    def test_text_too_short(self):
        example = {"text": "نص قصير"}
        valid, error = validate_cpt_example(example)
        assert valid is False
        assert "Text too short" in error

    def test_text_too_long(self):
        example = {"text": "أ" * 100001}
        valid, error = validate_cpt_example(example)
        assert valid is False
        assert "Text too long" in error

    def test_non_string_text(self):
        example = {"text": 123}
        valid, error = validate_cpt_example(example)
        assert valid is False
        assert "must be string" in error


class TestValidateSFTExample:
    """Tests for SFT example validation."""

    def test_valid_example(self):
        example = {
            "instruction": "اكتب قصيدة قصيرة",
            "input": "",
            "output": "في الريف حياة هادئة"
        }
        valid, error = validate_sft_example(example)
        assert valid is True
        assert error == ""

    def test_valid_with_input(self):
        example = {
            "instruction": "ترجم النص التالي",
            "input": "Hello world",
            "output": "مرحبا بالعالم"
        }
        valid, error = validate_sft_example(example)
        assert valid is True

    def test_missing_instruction(self):
        example = {"output": "جواب"}
        valid, error = validate_sft_example(example)
        assert valid is False
        assert "instruction" in error

    def test_missing_output(self):
        example = {"instruction": "سؤال"}
        valid, error = validate_sft_example(example)
        assert valid is False
        assert "output" in error

    def test_instruction_too_short(self):
        example = {"instruction": "أ", "output": "جواب طويل"}
        valid, error = validate_sft_example(example)
        assert valid is False
        assert "Instruction too short" in error

    def test_output_too_short(self):
        example = {"instruction": "سؤال طويل", "output": "أ"}
        valid, error = validate_sft_example(example)
        assert valid is False
        assert "Output too short" in error


class TestCheckDatasetSchema:
    """Tests for dataset schema checking."""

    def test_valid_schema(self):
        class MockDataset:
            column_names = ["text", "metadata"]
        
        result = check_dataset_schema(MockDataset(), ["text"])
        assert result is True

    def test_missing_columns(self):
        class MockDataset:
            column_names = ["text"]
        
        with pytest.raises(ValueError) as excinfo:
            check_dataset_schema(MockDataset(), ["text", "metadata"])
        assert "Missing columns" in str(excinfo.value)


class TestGetDatasetStats:
    """Tests for dataset statistics."""

    def test_empty_dataset(self):
        class MockDataset:
            def __iter__(self):
                return iter([])
        
        stats = get_dataset_stats(MockDataset())
        assert stats == {}

    def test_basic_stats(self):
        class MockDataset:
            def __iter__(self):
                return iter([
                    {"text": "نص قصير"},
                    {"text": "نص أطول قليلاً من الأول"},
                    {"text": "نص طويل جداً يحتوي على الكثير من الكلمات والحرف العربية"}
                ])
        
        stats = get_dataset_stats(MockDataset())
        assert stats["count"] == 3
        assert stats["min_length"] > 0
        assert stats["max_length"] > stats["min_length"]
