# Prepare Data from Hugging Face - Detailed Guide

## Overview

This guide covers the complete data preparation pipeline for Baligh-1.7B v0,
from downloading Hugging Face datasets to producing train-ready corpora.

## Pipeline Stages

1. **Download** - Stream/download from HF Hub
2. **Clean** - HTML removal, normalization, quality filtering
3. **Deduplicate** - Exact + near-duplicate removal
4. **Mix** - Combine datasets by configured ratios
5. **Format** - Tokenize and pack for CPT/SFT
6. **Validate** - Schema and quality checks
7. **Export** - Save as Parquet/Arrow with metadata
