# Data Flow Design

## Overview

This document describes the data flow through the Baligh-1.7B v0 training pipeline.

## Phase 1: Data Ingestion

### Source Datasets
- CPT: ArabicWeb24, ArabicText-Large, The Arabic Pile, OSCAR, mC4
- Islamic: hadith_datasets, quran_qa, quran_md
- SFT: CIDAR, evol-instruct-arabic, Gazelle, Summarization
- Eval: MMLU-Arabic, CIDAR eval, mr-tydi Arabic

### Loading Strategy
- Streaming for large datasets (ArabicWeb24, OSCAR, mC4, Arabic Pile)
- Full load for smaller datasets (CIDAR, evol-instruct, Islamic)
- Dataset registry with metadata (license, tokens, domain)

## Phase 2: Data Cleaning

### Cleaning Pipeline Steps
1. HTML/boilerplate removal
2. URL/email removal
3. Control character removal
4. Arabic normalization (alef variants, teh marbuta, ZWNJ)
5. Repeated punctuation fixing
6. Whitespace normalization
7. Language filtering (Arabic > 30%)
8. Quality filtering (length, repetition, punctuation ratio)
9. Near-deduplication (MinHash LSH, optional)

### Output Format
- Parquet with metadata: source, domain, license, quality, split, text_length, doc_id
- JSONL for smaller datasets
- Arrow for HF datasets compatibility

## Phase 3: Dataset Mixing

### CPT Mixing Ratios
- ArabicWeb24: 70%
- ArabicText-Large: 20%
- The Arabic Pile: 10%
- Islamic: Separate cycle

### SFT Mixing Ratios
- CIDAR: 40%
- evol-instruct-arabic: 35%
- Gazelle: 10%
- Summarization: 10%
- Islamic QA: 5%

### Interleaving Strategy
- interleave_datasets with probabilities
- Seed: 42 for reproducibility
- Stopping: first_exhausted

## Phase 4: Prompt Formatting

### CPT Formatting
- Simple text tokenization
- Labels = input_ids (causal LM)
- Packing enabled for efficiency

### SFT Formatting
- Chat template (Qwen3 ChatML format; valid-Jinja fallback installed for
  template-less base tokenizers)
- Messages: system -> user -> assistant
- Response-only loss masking
- Packing disabled (preserve conversation structure)

## Phase 5: Training Data Output

- Saved to disk as HF datasets
- CPT: data/train_ready/cpt/
- SFT: data/train_ready/sft/
- Eval: data/train_ready/eval/
