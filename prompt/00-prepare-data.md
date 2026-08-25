---
description: Prepare all CPT and SFT training data in one orchestrated pipeline — download, clean, deduplicate, mix, format, split.
agent: builder
model: ollama/qwen3.5:9b
---

# Goal

Run the existing `prepare_data.py` pipeline to produce Arrow-format train/eval splits ready for CPT and SFT training. The pipeline handles download, schema standardization, cleaning, deduplication, validation, mixing, formatting, and splitting.

# What the Pipeline Does (already implemented)

The `DataPreparator` class in `src/scripts/prepare_data.py` (399 lines) orchestrates:

1. **Load** — downloads from HuggingFace using `datasets.py` registry (15 datasets).
2. **Standardize** — maps all sources to canonical schemas: CPT → `{"text"}`, SFT → `{"instruction", "input", "output"}`.
3. **Clean** — 7-step `CleaningPipeline` (HTML/URL/email removal, Arabic normalization, whitespace collapse). Diacritics preserved for Islamic text.
4. **Deduplicate** — MinHash LSH over character 3-gram shingles (optional).
5. **Validate** — per-example schema + quality checks.
6. **Mix** — ratio-based interleaving via `DatasetMixer` (CPT: 70/20/10, SFT: 40/35/15/10).
7. **Format** — tokenize with ChatML template + response-only loss masking (SFT).
8. **Split** — train/eval split (1% for CPT, 2% for SFT).

# Steps to Execute

1. Open a terminal in the project root: `D:\AI\Projects\LLM\Baligh`.
2. Run the CPT pipeline:
   ```bash
   uv run python -m src.scripts.prepare_data --stage cpt --clean --deduplicate --output-dir data/train_ready
   ```
3. Wait for completion. Verify output exists: `data/train_ready/cpt/` (Arrow format with `train`/`eval` splits).
4. Run the SFT pipeline:
   ```bash
   uv run python -m src.scripts.prepare_data --stage sft --clean --output-dir data/train_ready
   ```
5. Wait for completion. Verify output exists: `data/train_ready/sft/` (Arrow format with `train`/`eval` splits).
6. For a quick smoke test with limited samples (fast validation):
   ```bash
   uv run python -m src.scripts.prepare_data --stage both --clean --max-samples 1000 --output-dir data/train_ready_test
   ```

# Verification

- Check `data/train_ready/cpt/metadata.json` — should list train/eval split sizes.
- Check `data/train_ready/sft/metadata.json` — should list train/eval split sizes.
- If any step fails, read the error message and fix before continuing.

# Constraints

- Do NOT write new scripts — use the existing `prepare_data.py`.
- Do NOT save data inside Git.
- Do NOT modify `src/` code unless there's a bug in the pipeline.
- Streaming datasets (arabicweb24, oscar_ar, mc4_ar, arabic_pile) will be materialized with `--no-streaming` or bounded by `--max-samples` for the smoke test.
