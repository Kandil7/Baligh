"""Tests for preference data handling (DPO)."""

import pytest
from datasets import Dataset

from baligh.data.preference import (
    get_preference_stats,
    load_preference_data,
    normalize_preference_schema,
    validate_preference_example,
)


@pytest.fixture
def canonical_rows():
    return [
        {
            "prompt": "ما هو التوحيد؟",
            "chosen": "التوحيد إفراد الله بالعبادة.",
            "rejected": "لا أعرف.",
        },
        {
            "prompt": "اذكر أركان الإسلام",
            "chosen": "خمسة أركان: الشهادتان والصلاة...",
            "rejected": "ثلاثة فقط.",
        },
    ]


@pytest.fixture
def alias_rows():
    return [
        {
            "instruction": "اشرح العدل",
            "chosen": "العدل وضع كل شيء في موضعه.",
            "rejected": "العدل نوع من الطعام.",
        },
    ]


class TestValidatePreferenceExample:
    def test_valid_canonical(self):
        example = {"prompt": "سؤال", "chosen": "جيد", "rejected": "سيئ"}
        assert validate_preference_example(example) is True

    def test_valid_instruction_alias(self):
        example = {"instruction": "سؤال", "chosen": "جيد", "rejected": "سيئ"}
        assert validate_preference_example(example) is True

    def test_missing_rejected(self):
        assert validate_preference_example({"prompt": "س", "chosen": "ج"}) is False

    def test_empty_chosen(self):
        example = {"prompt": "س", "chosen": "", "rejected": "سيئ"}
        assert validate_preference_example(example) is False


class TestNormalizePreferenceSchema:
    def test_canonical_untouched(self, canonical_rows):
        ds = Dataset.from_list(canonical_rows)
        result = normalize_preference_schema(ds)
        assert result.column_names == ["prompt", "chosen", "rejected"]

    def test_instruction_renamed_to_prompt(self, alias_rows):
        ds = Dataset.from_list(alias_rows)
        result = normalize_preference_schema(ds)
        assert "prompt" in result.column_names
        assert "instruction" not in result.column_names

    def test_invalid_schema_raises(self):
        ds = Dataset.from_list([{"question": "س", "good": "ج", "bad": "خ"}])
        with pytest.raises(ValueError, match="prompt"):
            normalize_preference_schema(ds)


class TestLoadPreferenceData:
    def test_load_jsonl(self, tmp_path, canonical_rows):
        import json

        path = tmp_path / "pref.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in canonical_rows),
            encoding="utf-8",
        )
        ds = load_preference_data(path)
        assert len(ds) == 2
        assert ds[0]["prompt"] == canonical_rows[0]["prompt"]

    def test_load_arrow_dir(self, tmp_path, canonical_rows):
        ds = Dataset.from_list(canonical_rows)
        ds.save_to_disk(str(tmp_path / "arrow_dir"))
        loaded = load_preference_data(tmp_path / "arrow_dir")
        assert len(loaded) == 2


class TestGetPreferenceStats:
    def test_snake_case_keys_and_values(self, canonical_rows):
        stats = get_preference_stats(Dataset.from_list(canonical_rows))
        assert stats["num_examples"] == 2
        assert "chosen_max_words" in stats
        assert "rejected_max_words" in stats
        assert "chosenMaxLength" not in stats
        assert stats["chosen_mean_words"] > 0
