# Benchmark Definitions

## MMLU-Arabic

- Source: FreedomIntelligence/MMLU_Arabic
- Task: Multiple-choice QA across subjects
- Split: test
- Metric: Accuracy
- Subjects: STEM, humanities, social sciences, etc.

## CIDAR-EVAL-100

- Source: arbml/CIDAR (test split)
- Task: Open-ended generation
- Metric: ROUGE-1, ROUGE-2, ROUGE-L
- Focus: Cultural relevance

## CIDAR-MCQ-100

- Source: arbml/CIDAR (test split)
- Task: Multiple-choice
- Metric: Accuracy
- Focus: Cultural knowledge

## mr-tydi Arabic

- Source: castorini/mr-tydi (arabic)
- Task: Retrieval-based QA
- Metric: F1, Exact Match
- Documents: Wikipedia passages

## Islamic QA Custom

- Source: Curated from hadith, quran, fiqh sources
- Task: Open-ended QA
- Metric: ROUGE-L, Exact Match, Human eval
- Categories: Fiqh, Hadith, Tafsir, Aqidah, Seerah

## Arabic Perplexity

- Source: Held-out ArabicWeb24 / ArabicText-Large
- Metric: Perplexity (exp of cross-entropy loss)
- Batch size: 8, stride: 512

## Long-Context Evaluation

- Prompts: 4K, 8K, 16K, 32K tokens
- Tasks: Summarization, QA, Key info extraction
- Metric: ROUGE, Accuracy, Faithfulness

## Formatting Tasks

- JSON extraction
- Markdown table generation
- Structured output
- Metric: JSON validity, Schema compliance

## Refusal/Uncertainty

- Prompts requiring I don't know
- Sensitive topics
- Metric: Refusal rate, Appropriate uncertainty
