# Baligh-1.5B v0

Arabic-first LLM with Islamic knowledge specialization, built on Qwen2.5-1.5B Base.

## Overview

Baligh-1.5B v0 is a lightweight Arabic language model (1.54B parameters) optimized for:
- **Formal Arabic (Fusha)** fluency and quality
- **Islamic knowledge** (Quran, Hadith, Fiqh, Tafsir)
- **Instruction following** in Arabic
- **Structured output** (JSON extraction)

## Architecture

- **Base Model**: Qwen2.5-1.5B (1.54B params, 28 layers, GQA, 32K context)
- **Training**: Continued Pretraining (CPT) + Supervised Fine-Tuning (SFT)
- **Method**: QLoRA 4-bit (r=16, alpha=16)
- **Tokenizer**: Qwen2.5 tokenizer (unchanged for v0)

## Training Pipeline

### Stage 1: Continued Pretraining (CPT)
- **Datasets**: ArabicWeb24 (70%), ArabicText-Large (20%), The Arabic Pile (10%)
- **Islamic Cycle**: Hadith collections, Quran QA, Quran texts
- **Tokens**: ~20-50B Arabic tokens
- **Config**: 

### Stage 2: Supervised Fine-Tuning (SFT)
- **Datasets**: CIDAR (40%), evol-instruct-arabic (35%), Gazelle (10%), Summarization (10%), Islamic QA (5%)
- **Examples**: ~100K-500K instruction examples
- **Config**: 

## Quick Start



## Docker



## Configuration

- **Base Config**: 
- **CPT Config**: 
- **SFT Config**: 
- **Eval Config**: 

## Evaluation

Benchmarks:
- MMLU-Arabic
- CIDAR (cultural relevance)
- mr-tydi Arabic (retrieval QA)
- Custom Islamic QA
- Arabic Perplexity
- Human evaluation (1-5 rubric)

## Release Artifacts

- : After CPT
- : After SFT
- Quantized variants: GGUF, AWQ, GPTQ

## License

Apache 2.0

## Citation


