# Evaluation Framework

## Overview

Multi-level evaluation: automated metrics + human evaluation.

## Automated Evaluation

### Language Quality
- Perplexity on Arabic held-out set
- Morphology and punctuation coverage
- Encoding/diacritics/language mixing errors

### Instruction Following
- Summarization
- Reformulation
- Entity extraction
- Classification
- Structured output (JSON)

### Islamic Domain
- Knowledge questions (fiqh, hadith, tafsir)
- Terminology accuracy
- Hallucination rate on precise matters
- Uncertainty expression ("I don')

### Safety & Quality
- Overconfidence detection
- Fatwa vs. transmission distinction
- Response length appropriateness
- Arabic style consistency

## Benchmark Datasets

| Dataset | Purpose | Metric |
|---------|---------|--------|
| MMLU-Arabic | General QA | Accuracy |
| CIDAR-EVAL-100 | Cultural relevance | ROUGE-L |
| CIDAR-MCQ-100 | Cultural MCQ | Accuracy |
| mr-tydi Arabic | Retrieval QA | F1/EM |
| Islamic QA Custom | Domain knowledge | ROUGE-L, EM |
| Arabic Perplexity | Language modeling | PPL |

## Human Evaluation

### Rubric (1-5 scale)
| Criterion | Description |
|-----------|-------------|
| Correctness | Factual accuracy |
| Clarity | Readability of Arabic |
| Arabic Quality | Formal Arabic (fusha) |
| Usefulness | Helpfulness for query |
| Faithfulness | Faithfulness to knowledge |

### Process
- 100-500 samples across domains
- Multiple annotators
- Inter-annotator agreement (Krippendorff alpha)
- Blind evaluation (model identity hidden)

## Test Suite

- Fixed eval set: 100-500 examples
- Never enters training
- Categories: Arabic language, Islamic QA, long-context, formatting, refusal
- Version-controlled for regression testing

## Running Evaluation


