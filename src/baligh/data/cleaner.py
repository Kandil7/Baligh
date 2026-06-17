"""Data cleaning pipeline for Baligh-1.5B v0."""

import html
import re

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

# Compiled regex patterns for efficiency
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MULTIPLE_SPACES_RE = re.compile(r"\s+")
MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
REPEATED_PUNCT_RE = re.compile(r"([.!?]){2,}")
ARABIC_DIACRITICS_RE = re.compile(r"[ً-ٰٟـ]")
ZWNJ_RE = re.compile(r"[\u200C\u200D]")
CONTROL_CHARS_RE = re.compile(r"[--]")


def remove_html(text):
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def remove_urls(text):
    return URL_RE.sub(" ", text)


def remove_emails(text):
    return EMAIL_RE.sub(" ", text)


def normalize_whitespace(text):
    text = MULTIPLE_SPACES_RE.sub(" ", text)
    text = MULTIPLE_NEWLINES_RE.sub("", text)
    return text.strip()


def normalize_arabic(text):
    # Alef normalization: alef with hamza, alef with madda, alef with hamza below -> plain alef
    text = text.replace("\u0623", "\u0627")  # alef with hamza above -> alef
    text = text.replace("\u0622", "\u0627")  # alef with madda -> alef
    text = text.replace("\u0625", "\u0627")  # alef with hamza below -> alef
    # Yaa normalization: tail yaa -> standard yaa
    text = text.replace("\u0649", "\u064a")  # tail yaa -> standard yaa
    # Remove tatweel
    text = text.replace("\u0640", "")
    # Remove ZWNJ and ZWJ
    text = ZWNJ_RE.sub("", text)
    return text


def remove_control_chars(text):
    return CONTROL_CHARS_RE.sub("", text)


def fix_repeated_punctuation(text):
    return REPEATED_PUNCT_RE.sub(r"", text)


def is_arabic(text, threshold=0.3):
    if not text:
        return False
    arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    return arabic_chars / len(text) >= threshold


def is_low_quality(text, min_length=50, max_length=100000):
    if not text or len(text) < min_length or len(text) > max_length:
        return True
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True
    punct_ratio = sum(1 for c in text if c in ".,!?;:") / len(text)
    if punct_ratio > 0.1:
        return True
    return False


class CleaningPipeline:
    def __init__(
        self,
        steps=None,
        language_filter="arabic",
        quality_filter=True,
        min_length=50,
        max_length=100000,
    ):
        self.steps = steps or [
            remove_html,
            remove_urls,
            remove_emails,
            remove_control_chars,
            normalize_arabic,
            fix_repeated_punctuation,
            normalize_whitespace,
        ]
        self.language_filter = language_filter
        self.quality_filter = quality_filter
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text):
        if not text:
            return None
        for step in self.steps:
            text = step(text)
            if not text:
                return None
        if self.language_filter == "arabic" and not is_arabic(text):
            return None
        if self.quality_filter and is_low_quality(text, self.min_length, self.max_length):
            return None
        return text

    def clean_batch(self, texts):
        cleaned = []
        for text in texts:
            result = self.clean(text)
            if result:
                cleaned.append(result)
        return cleaned

    def __call__(self, batch):
        if "text" in batch:
            cleaned = self.clean_batch(batch["text"])
            return {"text": cleaned}
        return batch


def get_cleaning_pipeline(language_filter="arabic", quality_filter=True):
    return CleaningPipeline(language_filter=language_filter, quality_filter=quality_filter)


def deduplicate_dataset(dataset, text_column="text", num_perm=128, threshold=0.7):
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("datasketch not installed, skipping deduplication")
        return dataset
    logger.info("Deduplicating dataset with %d examples" % len(dataset))
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices = []
    for idx, example in enumerate(dataset):
        text = example[text_column]
        if not text:
            continue
        m = MinHash(num_perm=num_perm)
        words = text.split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i : i + 3])
            m.update(shingle.encode("utf-8"))
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            keep_indices.append(idx)
    logger.info("Kept %d / %d examples after deduplication" % (len(keep_indices), len(dataset)))
    return dataset.select(keep_indices)
