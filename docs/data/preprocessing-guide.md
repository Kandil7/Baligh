# Data Preprocessing Guide

## Overview

This guide describes the data cleaning and preprocessing pipeline for Baligh-1.7B v0.

## Cleaning Pipeline Steps

### 1. HTML/Boilerplate Removal
- Regex-based HTML tag removal
- html.unescape for entities

### 2. URL/Email Removal
- Regex patterns for http(s):// and www.
- Email pattern matching

### 3. Control Character Removal
- Remove ASCII control chars (0-8, 11, 12, 14-31, 127)

### 4. Arabic Normalization
- Alef variants -> alef (آ،أ،إ -> ا)
- Teh marbuta -> heh (ة -> ه)
- Remove ZWNJ/ZWJ (‌, ‍)
- Diacritics: preserved by default (configurable)

### 5. Punctuation Fixing
- Collapse repeated punctuation (??? -> ?)

### 6. Whitespace Normalization
- Multiple spaces -> single space
- Multiple newlines -> double newline
- Trim leading/trailing

### 7. Language Filtering
- Arabic character ratio >= 30%
- Unicode range: ؀-ۿ

### 8. Quality Filtering
- Min length: 50 chars
- Max length: 100,000 chars
- Unique word ratio >= 30%
- Punctuation ratio <= 10%

### 9. Near-Deduplication (Optional)
- MinHash LSH with datasketch
- 128 permutations, Jaccard threshold 0.7
- Word-level 3-gram shingles

## Running the Pipeline



## Output Structure

data/train_ready/
├── cpt/          # Mixed CPT dataset
├── sft/          # Mixed SFT dataset
└── eval/         # Evaluation datasets

Each saved as Hugging Face dataset (Arrow format) with metadata columns.
