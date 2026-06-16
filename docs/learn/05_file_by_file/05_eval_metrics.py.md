# eval metrics.py - Evaluation Metrics

**Path**: `src/baligh/evaluation/metrics.py` (48 lines)

## Purpose

Standard NLP evaluation metrics for Arabic text generation.

---

## compute_perplexity (lines 11-26)

Language modeling perplexity over a dataloader. Same implementation as training/metrics.py.

---

## compute_rouge (lines 28-35)

ROUGE-1, ROUGE-2, ROUGE-L F-measure using rouge_score library with stemmer enabled.

---

## compute_bleu (lines 37-40)

Corpus-level BLEU score using sacrebleu.

---

## compute_bert_score (lines 42-44)

BERTScore F1 using the multilingual model with lang="ar" for Arabic.

---

## compute_exact_match (lines 46-48)

Exact string match ratio after stripping whitespace.
