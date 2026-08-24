"""Tests for the unified dataset loader (single registry contract)."""

from unittest.mock import MagicMock, patch

import pytest

from baligh.data.datasets import DATASETS
from baligh.data.loader import list_registered_names, load_dataset_by_name


class TestRegistryUnification:
    def test_all_registered_types_loadable_by_name(self):
        """Regression: v0.1 had a second 3-entry registry, so every SFT/eval
        dataset raised ValueError at load time."""
        for name in DATASETS:
            assert list_registered_names().count(name) == 1

    def test_sft_names_resolvable(self):
        for name in ("cidar", "evol_instruct_arabic", "gazelle", "summarization"):
            assert name in DATASETS

    def test_unknown_name_raises_with_guidance(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_by_name("not_a_real_dataset")


class TestLoadDatasetByName:
    @patch("baligh.data.loader.load_dataset")
    def test_forwards_only_hf_kwargs(self, mock_load):
        """Metadata keys (license/tokens/domain) must never leak into
        load_dataset(**kwargs) — v0.1 splatted them and crashed."""
        mock_load.return_value = MagicMock()
        load_dataset_by_name("arabicweb24")
        kwargs = mock_load.call_args[1]
        for forbidden in ("text_column", "license", "tokens", "domain", "type"):
            assert forbidden not in kwargs
        assert kwargs["streaming"] is True

    @patch("baligh.data.loader.load_dataset")
    def test_split_and_streaming_overrides(self, mock_load):
        mock_load.return_value = MagicMock()
        load_dataset_by_name("cidar", split="test", streaming=True)
        kwargs = mock_load.call_args[1]
        assert kwargs["split"] == "test"
        assert kwargs["streaming"] is True


class TestStandardizers:
    def test_standardize_sft_maps_summarization_columns(self):
        """text->instruction, summary->output mapping must be honored."""
        from baligh.data.datasets import standardize_sft_dataset

        rows = [{"text": "مقال عن القاهرة", "summary": "مقال", "extra": "x"}]

        ds = MagicMock()
        ds.column_names = ["text", "summary", "extra"]
        mapped_rows = []

        def fake_map(fn, remove_columns=None, desc=None):
            out = [fn(r) for r in rows]
            mapped_rows.extend(out)
            result = MagicMock()
            result.column_names = list(out[0].keys())
            return result

        ds.map = fake_map
        standardized = standardize_sft_dataset(ds, "summarization")
        row = mapped_rows[0]
        assert row["instruction"] == "مقال عن القاهرة"
        assert row["output"] == "مقال"
        assert set(standardized.column_names) >= {"instruction", "input", "output"}

    def test_standardize_cpt_projects_text_column(self):
        """quran_qa uses 'context' as its text source; must map to 'text'."""
        from baligh.data.datasets import standardize_cpt_dataset

        rows = [{"context": "آية", "question": "سؤال", "answer": "جواب"}]

        ds = MagicMock()
        ds.column_names = ["context", "question", "answer"]
        mapped_rows = []

        def fake_map(fn, desc=None):
            out = [fn(r) for r in rows]
            mapped_rows.extend(out)
            result = MagicMock()
            result.column_names = list(out[0].keys())
            return result

        ds.map = fake_map
        standardize_cpt_dataset(ds, "quran_qa")
        assert mapped_rows[0]["text"] == "آية"
