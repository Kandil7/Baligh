"""Evaluation metrics for Baligh-1.7B v0 — Arabic-aware implementations.

Why custom ROUGE: ``rouge_score``'s default tokenizer lowercases and then
applies ``[a-z0-9]+`` — every Arabic character is stripped, producing empty
token lists and ROUGE ~= 0 regardless of output quality. We tokenize with
an Arabic-inclusive pattern and score standard clipped n-gram / LCS F1
directly, so numbers reflect the language this project exists for.

BLEU uses sacrebleu's dedicated Arabic tokenizer (``tokenize="ar"``);
the default 13a under-segments Arabic morphology.

Exact match normalizes both sides (alef/yaa variants, tatweel, diacritics,
punctuation, digit forms, casefold) so orthographic noise doesn't mask
semantic equality.
"""

import re
from collections import Counter
from typing import Any

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

# Arabic blocks + Latin + Western/Eastern-Arabic digits
_SCORING_TOKEN_RE = re.compile(
    r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"
    r"0-9\u0660-\u0669a-zA-Z]+"
)

_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_PUNCT_STRIP = str.maketrans("", "", ".,!?;:،؛؟…\"'()[]{}«»")


def tokenize_for_scoring(text: str) -> list[str]:
    """Tokenize Arabic/Latin text for metric computation."""
    return _SCORING_TOKEN_RE.findall(text or "")


def normalize_for_match(text: str) -> str:
    """Canonical form used by exact match.

    Applies the project's Arabic normalization (alef/yaa unification,
    tatweel removal), strips tashkeel and punctuation, folds Arabic-Indic
    digits to Western, and casefolds.
    """
    from baligh.data.cleaner import normalize_arabic

    text = normalize_arabic(text or "")
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(_PUNCT_STRIP)
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return " ".join(text.split()).casefold()


def _clipped_ngram_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((pred_ngrams & ref_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def _lcs_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """ROUGE-1/2/L F1 over Arabic-aware tokens. Length-mismatched inputs raise."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions/references length mismatch: {len(predictions)} vs {len(references)}"
        )
    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references, strict=True):
        p_tokens = tokenize_for_scoring(pred)
        r_tokens = tokenize_for_scoring(ref)
        scores["rouge1"].append(_clipped_ngram_f1(p_tokens, r_tokens, 1))
        scores["rouge2"].append(_clipped_ngram_f1(p_tokens, r_tokens, 2))
        scores["rougeL"].append(_lcs_f1(p_tokens, r_tokens))
    return {k: float(sum(v) / len(v)) if v else 0.0 for k, v in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus BLEU with sacrebleu's Arabic-aware tokenizer."""
    from sacrebleu import corpus_bleu

    refs = [[r] for r in references]
    bleu = corpus_bleu(list(predictions), refs, tokenize="ar")
    return bleu.score


def compute_bert_score(
    predictions: list[str], references: list[str], lang: str = "ar"
) -> dict[str, float]:
    """BERTScore (multilingual encoder handles Arabic natively)."""
    from bert_score import score as bert_score_fn

    precision, recall, f1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Normalized exact match (see normalize_for_match)."""
    if not predictions:
        return 0.0
    matches = sum(
        1
        for p, r in zip(predictions, references, strict=False)
        if normalize_for_match(p) == normalize_for_match(r)
    )
    return matches / len(predictions)


def compute_perplexity(model: Any, dataloader: Any, device: Any) -> float:
    """Token-weighted perplexity (delegates to the shared implementation)."""
    from baligh.training.metrics import compute_perplexity as _impl

    return float(_impl(model, dataloader, device))
