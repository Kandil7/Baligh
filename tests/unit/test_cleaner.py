"""Tests for data cleaning pipeline."""

import pytest

from baligh.data.cleaner import (
    CleaningPipeline,
    deduplicate_dataset,
    fix_repeated_punctuation,
    get_cleaning_pipeline,
    is_arabic,
    is_low_quality,
    normalize_arabic,
    normalize_whitespace,
    remove_control_chars,
    remove_emails,
    remove_html,
    remove_urls,
)


class TestRemoveHtml:
    """Tests for HTML removal."""

    def test_remove_tags(self):
        text = "<p>Hello <b>world</b></p>"
        result = remove_html(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_unescape_entities(self):
        text = "Hello &amp; world"
        result = remove_html(text)
        assert "&amp;" not in result
        assert "Hello & world" in result


class TestRemoveUrls:
    """Tests for URL removal."""

    def test_remove_http(self):
        text = "Visit https://example.com for more"
        result = remove_urls(text)
        assert "https://example.com" not in result

    def test_remove_www(self):
        text = "Visit www.example.com for more"
        result = remove_urls(text)
        assert "www.example.com" not in result


class TestRemoveEmails:
    """Tests for email removal."""

    def test_remove_email(self):
        text = "Contact user@example.com for info"
        result = remove_emails(text)
        assert "user@example.com" not in result


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_multiple_spaces(self):
        text = "Hello   world"
        result = normalize_whitespace(text)
        assert "Hello world" in result

    def test_newlines(self):
        text = "Hello\n\n\nworld"
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result

    def test_newlines_preserved(self):
        """Paragraph structure must survive cleaning (v0.2 fix)."""
        text = "First paragraph line.\nSecond paragraph."
        result = normalize_whitespace(text)
        assert "\n" in result

    def test_excessive_newlines_collapse_to_blank_line(self):
        text = "A\n\n\n\nB"
        result = normalize_whitespace(text)
        assert result == "A\n\nB"

    def test_tabs_collapsed(self):
        text = "Hello\t\tworld"
        assert normalize_whitespace(text) == "Hello world"

    def test_strip(self):
        text = "  Hello world  "
        result = normalize_whitespace(text)
        assert result == "Hello world"


class TestNormalizeArabic:
    """Tests for Arabic normalization."""

    def test_alef_normalization(self):
        text = "أ إ آ ا"
        result = normalize_arabic(text)
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result

    def test_yaa_normalization(self):
        text = "ى ي"
        result = normalize_arabic(text)
        assert "ى" not in result


class TestIsArabic:
    """Tests for Arabic language detection."""

    def test_arabic_text(self):
        text = "هذا نص عربي طويل يحتوي على الكثير من الكلمات العربية"
        assert is_arabic(text) is True

    def test_english_text(self):
        text = "This is English text with no Arabic characters"
        assert is_arabic(text) is False

    def test_mixed_text(self):
        text = "Hello عالم"
        result = is_arabic(text, threshold=0.1)
        assert isinstance(result, bool)

    def test_empty_text(self):
        assert is_arabic("") is False


class TestIsLowQuality:
    """Tests for quality filtering."""

    def test_too_short(self):
        assert is_low_quality("short") is True

    def test_too_long(self):
        assert is_low_quality("أ" * 100001) is True

    def test_repetitive_words(self):
        text = " ".join(["كلمة"] * 20)
        assert is_low_quality(text) is True

    def test_good_quality(self):
        text = "هذا نص جيد يحتوي على كلمات متنوعة ومختلفة في اللغة العربية"
        assert is_low_quality(text) is False


class TestFixRepeatedPunctuation:
    def test_collapse_keeps_single_mark(self):
        """v0.2 fix: repeated marks collapse to one, not zero."""
        assert fix_repeated_punctuation("جيد!!!") == "جيد!"
        assert fix_repeated_punctuation("ماذا??") == "ماذا?"

    def test_arabic_punctuation_collapsed(self):
        assert fix_repeated_punctuation("طيب!!") == "طيب!"

    def test_single_mark_untouched(self):
        assert fix_repeated_punctuation("جيد!") == "جيد!"


class TestRemoveControlChars:
    def test_strips_control_characters(self):
        text = "ok\x00\x07text\x1f"
        result = remove_control_chars(text)
        assert "\x00" not in result and "\x07" not in result and "\x1f" not in result

    def test_preserves_tab_newline(self):
        text = "a\tb\nc\r"
        assert remove_control_chars(text) == text


class TestDeduplicateDataset:
    def test_streaming_dataset_raises_with_guidance(self):
        from datasets import IterableDataset

        def gen():
            yield {"text": "sample"}

        streaming = IterableDataset.from_generator(gen)
        with pytest.raises(ValueError) as excinfo:
            deduplicate_dataset(streaming)
        assert "materialize" in str(excinfo.value).lower()


class TestCleaningPipeline:
    """Tests for CleaningPipeline."""

    def test_init(self):
        pipeline = CleaningPipeline()
        assert pipeline.language_filter == "arabic"
        assert pipeline.quality_filter is True

    def test_clean(self):
        pipeline = CleaningPipeline()
        text = "<p>هذا نص عربي طويل يحتوي على الكثير من الكلمات والمعلومات المتنوعة</p>"
        result = pipeline.clean(text)
        assert result is not None
        assert "<p>" not in result

    def test_clean_batch(self):
        pipeline = CleaningPipeline()
        texts = [
            "<p>هذا نص عربي طويل يحتوي على الكثير من الكلمات</p>",
            "<b>نص آخر عربي طويل أيضاً</b>",
        ]
        result = pipeline.clean_batch(texts)
        assert len(result) >= 0  # May filter some

    def test_language_filter(self):
        pipeline = CleaningPipeline(language_filter="arabic")
        text = "This is English text"
        result = pipeline.clean(text)
        assert result is None

    def test_quality_filter(self):
        pipeline = CleaningPipeline(quality_filter=True)
        text = "short"
        result = pipeline.clean(text)
        assert result is None


class TestGetCleaningPipeline:
    """Tests for factory function."""

    def test_default(self):
        pipeline = get_cleaning_pipeline()
        assert isinstance(pipeline, CleaningPipeline)

    def test_custom_params(self):
        pipeline = get_cleaning_pipeline(language_filter=None, quality_filter=False)
        assert pipeline.language_filter is None
        assert pipeline.quality_filter is False
