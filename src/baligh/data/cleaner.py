"""Data cleaning pipeline for Baligh-1.7B v0.

Design notes:
- Whitespace normalization collapses runs of spaces/tabs but PRESERVES
  newlines: paragraph and verse boundaries carry signal for long-document
  behavior and Quran/Hadith text. Runs of 3+ newlines collapse to exactly 2.
- Repeated punctuation is collapsed to a single mark, not deleted
  ("جيد!!!" -> "جيد!").
- Diacritics (tashkeel) are intentionally NOT stripped anywhere: Islamic
  source text depends on them for correctness.
"""

import html
import re
from typing import Any

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

# Compiled regex patterns for efficiency
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Any horizontal whitespace run (spaces/tabs, NOT newlines) -> single space
MULTIPLE_SPACES_RE = re.compile(r"[^\S\n]+")
# 3+ consecutive newlines -> exactly one blank line (paragraph separator)
MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
# Repeated sentence punctuation -> single mark (English + Arabic marks)
REPEATED_PUNCT_RE = re.compile(r"([.!?،؛؟…])\1+")
ZWNJ_RE = re.compile(r"[\u200c\u200d]")
# Control characters except tab (\x09), newline (\x0a), carriage return (\x0d)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Punctuation used by the low-quality heuristic (English + Arabic)
_PUNCTUATION_CHARS = set(".,!?;:،؛؟…")


def remove_html(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    return EMAIL_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace and excessive blank lines.

    Newlines are preserved so paragraph/document structure survives cleaning;
    only runs of 3+ newlines are collapsed to a single blank line.
    """
    text = MULTIPLE_SPACES_RE.sub(" ", text)
    text = MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_arabic(text: str) -> str:
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


def remove_control_chars(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)


def fix_repeated_punctuation(text: str) -> str:
    """Collapse repeated punctuation runs to a single mark."""
    return REPEATED_PUNCT_RE.sub(r"\1", text)


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    if not text:
        return False
    arabic_chars = sum(
        1
        for c in text
        if "\u0600" <= c <= "\u06ff"  # Arabic block
        or "\u0750" <= c <= "\u077f"  # Arabic Supplement
        or "\u08a0" <= c <= "\u08ff"  # Arabic Extended-A
        or "\ufb50" <= c <= "\ufdff"  # Arabic Presentation Forms-A
        or "\ufe70" <= c <= "\ufeff"  # Arabic Presentation Forms-B
    )
    return arabic_chars / len(text) >= threshold


def is_low_quality(text: str | None, min_length: int = 50, max_length: int = 100000) -> bool:
    if not text or len(text) < min_length or len(text) > max_length:
        return True
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True
    punct_count = sum(1 for c in text if c in _PUNCTUATION_CHARS)
    return punct_ratio(punct_count, len(text)) > 0.1


def punct_ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


class CleaningPipeline:
    def __init__(
        self,
        steps: list | None = None,
        language_filter: str | None = "arabic",
        quality_filter: bool = True,
        min_length: int = 50,
        max_length: int = 100000,
    ) -> None:
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

    def clean(self, text: str | None) -> str | None:
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

    def sanitize(self, text: str | None) -> str:
        """Run cleaning steps WITHOUT dropping the row.

        Used for structured columns (instruction/input/output) where removing
        one field but not its siblings corrupts table alignment. Returns ""
        for rows that fail the language gate; quality filtering is skipped
        because short answers are legitimate there.
        """
        if not text:
            return ""
        for step in self.steps:
            text = step(text)
            if not text:
                return ""
        if self.language_filter == "arabic" and not is_arabic(text, threshold=0.1):
            # Keep mixed-language instruction fields (e.g. code snippets),
            # drop only clearly non-Arabic content.
            logger.debug("Sanitize: non-Arabic content blanked")
            return ""
        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        cleaned = []
        for text in texts:
            result = self.clean(text)
            if result:
                cleaned.append(result)
        return cleaned

    def __call__(self, batch: dict) -> dict:
        if "text" in batch:
            cleaned = self.clean_batch(batch["text"])
            return {"text": cleaned}
        return batch


def get_cleaning_pipeline(
    language_filter: str | None = "arabic", quality_filter: bool = True
) -> CleaningPipeline:
    return CleaningPipeline(language_filter=language_filter, quality_filter=quality_filter)


def deduplicate_dataset(
    dataset: Any,
    text_column: str = "text",
    num_perm: int = 128,
    threshold: float = 0.7,
) -> Any:
    """Near-duplicate removal via MinHash LSH over character 3-gram shingles.

    Only works on materialized (map-style) datasets. Streaming datasets must
    be materialized first — call this AFTER `.take(n)` + `Dataset.from_list`,
    or run with --no-streaming.
    """
    try:
        from datasets import Dataset, IterableDataset  # type: ignore
    except ImportError:  # pragma: no cover - datasets is a hard dep in practice
        logger.warning("datasets not installed, skipping deduplication")
        return dataset

    if isinstance(dataset, IterableDataset):
        raise ValueError(
            "deduplicate_dataset requires a materialized dataset. "
            "Streaming IterableDataset does not support len()/select(). "
            "Materialize a bounded sample first (e.g. Dataset.from_list(list(ds.take(N)))) "
            "or prepare data with --no-streaming."
        )

    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except ImportError:
        logger.warning("datasketch not installed, skipping deduplication")
        return dataset

    logger.info(f"Deduplicating dataset with {len(dataset)} examples")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices = []
    for idx, example in enumerate(dataset):
        text = example[text_column]
        if not text:
            continue
        m = MinHash(num_perm=num_perm)
        words = text.split()
        # Documents too short for 3-gram shingles keep their identity hash;
        # without this they all produce empty signatures and collapse together.
        if len(words) < 3:
            m.update(f"__short__:{text.strip()}".encode())
        else:
            for i in range(len(words) - 2):
                shingle = " ".join(words[i : i + 3])
                m.update(shingle.encode("utf-8"))
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            keep_indices.append(idx)
    logger.info(f"Kept {len(keep_indices)} / {len(dataset)} examples after deduplication")
    return dataset.select(keep_indices) if isinstance(dataset, Dataset) else dataset
