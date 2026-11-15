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
        for name in ("cidar", "evol_instruct_arabic", "islamweb_fatwa", "summarization"):
            assert name in DATASETS

    def test_removed_gazelle_is_gone(self):
        """Gazelle/arabic-writing 404s and the real repo is a 200-row
        benchmark; at 0.15 weight it would cap the whole mix at ~1,300 rows."""
        assert "gazelle" not in DATASETS

    def test_islamweb_is_a_local_split_mapped_source(self):
        cfg = DATASETS["islamweb_fatwa"]
        assert isinstance(cfg["path"], dict)
        assert set(cfg["path"]) == {"train", "eval"}
        assert cfg["type"] == "sft"

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


class TestLocalSources:
    def test_mapping_path_is_always_local(self):
        from baligh.data.loader import _is_local_source

        assert _is_local_source({"train": "a.jsonl", "eval": "b.jsonl"}) is True

    def test_hub_id_with_json_suffix_is_not_local(self):
        """A repo id must never be treated as a file even if it ends in
        a JSON-like suffix."""
        from baligh.data.loader import _is_local_source

        assert _is_local_source("org/some_dataset") is False
        assert _is_local_source("org/dataset.json") is False

    def test_existing_jsonl_path_is_local(self, tmp_path):
        from baligh.data.loader import _is_local_source

        f = tmp_path / "rows.jsonl"
        f.write_text('{"instruction": "i", "input": "", "output": "o"}\n', encoding="utf-8")
        assert _is_local_source(str(f)) is True
        assert _is_local_source(str(tmp_path / "missing.jsonl")) is False

    @patch("baligh.data.loader.load_dataset")
    def test_local_mapping_selects_the_requested_split(self, mock_load):
        from baligh.data.loader import _load_local

        mock_load.return_value = MagicMock()
        _load_local(
            {"train": "tr.jsonl", "eval": "ev.jsonl"},
            split="eval",
            streaming=False,
        )
        kwargs = mock_load.call_args[1]
        assert kwargs["data_files"] == {"eval": "ev.jsonl"}
        assert kwargs["split"] == "eval"

    @patch("baligh.data.loader.load_dataset")
    def test_local_mapping_rejects_unknown_split(self, mock_load):
        from baligh.data.loader import _load_local

        with pytest.raises(ValueError, match="no split 'test'"):
            _load_local({"train": "tr.jsonl", "eval": "ev.jsonl"}, split="test", streaming=False)
        mock_load.assert_not_called()


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

        def fake_map(fn, remove_columns=None, desc=None):
            out = [fn(r) for r in rows]
            mapped_rows.extend(out)
            result = MagicMock()
            result.column_names = list(out[0].keys())
            return result

        ds.map = fake_map
        standardize_cpt_dataset(ds, "quran_qa")
        assert mapped_rows[0]["text"] == "آية"

    def test_standardize_sharegpt_takes_first_human_and_last_gpt(self):
        """evol-instruct-arabic ships ShareGPT turns, not flat columns; the
        intermediate turns must be dropped, not concatenated into one answer."""
        from baligh.data.datasets import standardize_sft_dataset

        rows = [
            {
                "conversations": [
                    {"from": "human", "value": "ما حكم الصلاة؟"},
                    {"from": "gpt", "value": "جواب أول"},
                    {"from": "human", "value": "وسؤال ثانٍ"},
                    {"from": "gpt", "value": "الجواب النهائي"},
                ]
            }
        ]

        ds = MagicMock()
        ds.column_names = ["conversations"]
        mapped = []

        def fake_map(fn, remove_columns=None, desc=None):
            out = [fn(r) for r in rows]
            mapped.extend(out)
            result = MagicMock()
            result.column_names = list(out[0].keys())
            return result

        ds.map = fake_map
        standardize_sft_dataset(ds, "evol_instruct_arabic")
        row = mapped[0]
        assert row["instruction"] == "ما حكم الصلاة؟"
        assert row["output"] == "الجواب النهائي"
        assert set(row) == {"instruction", "input", "output"}

    def test_sharegpt_sources_declare_the_format_flag(self):
        assert DATASETS["evol_instruct_arabic"].get("format") == "sharegpt"
        assert "format" not in DATASETS["cidar"]
