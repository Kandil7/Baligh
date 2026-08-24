"""Tests for evaluation metrics - Arabic-aware behavior."""

import pytest

from baligh.evaluation.metrics import (
    compute_exact_match,
    compute_rouge,
    normalize_for_match,
    tokenize_for_scoring,
)


class TestTokenizeForScoring:
    def test_arabic_tokens_survive(self):
        tokens = tokenize_for_scoring("السلام عليكم ورحمة الله")
        assert len(tokens) == 4
        assert "السلام" in tokens

    def test_english_tokens(self):
        tokens = tokenize_for_scoring("Hello world 123")
        assert tokens == ["Hello", "world", "123"]

    def test_mixed(self):
        tokens = tokenize_for_scoring("قال: Hello!")
        assert "قال" in tokens
        assert "Hello" in tokens

    def test_empty(self):
        assert tokenize_for_scoring("") == []
        assert tokenize_for_scoring(None) == []


class TestNormalizeForMatch:
    def test_diacritics_stripped(self):
        assert normalize_for_match("مُحَمَّد") == normalize_for_match("محمد")

    def test_alef_variants_unified(self):
        assert normalize_for_match("أحمد") == normalize_for_match("احمد")

    def test_punctuation_ignored(self):
        assert normalize_for_match("نعم.") == normalize_for_match("نعم")

    def test_arabic_digits_folded(self):
        assert normalize_for_match("٢٠٢٦") == normalize_for_match("2026")

    def test_casefolded(self):
        assert normalize_for_match("Hello") == normalize_for_match("hello")


class TestComputeExactMatch:
    def test_exact_match(self):
        predictions = ["hello", "world"]
        references = ["hello", "world"]
        assert compute_exact_match(predictions, references) == 1.0

    def test_no_match(self):
        predictions = ["hello", "world"]
        references = ["foo", "bar"]
        assert compute_exact_match(predictions, references) == 0.0

    def test_partial_match(self):
        predictions = ["hello", "world"]
        references = ["hello", "foo"]
        assert compute_exact_match(predictions, references) == 0.5

    def test_empty(self):
        assert compute_exact_match([], []) == 0.0

    def test_case_insensitive_after_normalization(self):
        # Deliberate behavior change v0.2: matching is normalized, so case
        # differences no longer mask semantic equality.
        predictions = ["Hello"]
        references = ["hello"]
        assert compute_exact_match(predictions, references) == 1.0

    def test_whitespace_handling(self):
        predictions = ["  hello  "]
        references = ["hello"]
        assert compute_exact_match(predictions, references) == 1.0

    def test_arabic_normalization_enables_match(self):
        """The whole point: diacritic/alef noise must not mask a correct answer."""
        predictions = ["مُحَمَّد"]
        references = ["محمد"]
        assert compute_exact_match(predictions, references) == 1.0


class TestComputeRouge:
    def test_identical_texts_score_one(self):
        text = "قال رسول الله إن الصوم جنة"
        scores = compute_rouge([text], [text])
        for key in ("rouge1", "rouge2", "rougeL"):
            assert scores[key] == 1.0, f"{key} should be 1.0 for identical Arabic text"

    def test_disjoint_texts_score_zero(self):
        scores = compute_rouge(["قط قريب"], ["كلب بعيد"])
        for key in ("rouge1", "rouge2", "rougeL"):
            assert scores[key] == 0.0

    def test_arabic_not_zeroed_out(self):
        """Regression guard: rouge_score's [a-z0-9]+ tokenizer used to strip
        every Arabic character and return ~0 regardless of quality.

        pred adds two tokens to a three-token reference; LCS F1 = 0.75.
        """
        pred = "الصيام وسيلة للتقوى في الإسلام"
        ref = "الصيام وسيلة للتقوى"
        scores = compute_rouge([pred], [ref])
        assert scores["rougeL"] == pytest.approx(0.75)

    def test_length_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError):
            compute_rouge(["a b c"], ["a b", "c d"])
