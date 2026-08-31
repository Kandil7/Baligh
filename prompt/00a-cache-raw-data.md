---
description: Cache raw registry datasets to D:\AI\Datasets\baligh\ as JSONL so pipeline reruns never re-download web-scale corpora.
agent: arabic-data-engineer
model: ollama/qwen3.5:9b-32k
---

# Goal

Build an immutable raw cache of every registered dataset before running `prepare_data.py`. This is an OPTIONAL offline layer: the training pipeline still streams from HuggingFace by default; this cache exists so reruns after network failure cost zero bandwidth.

# What Already Exists

`src/scripts/cache_raw_data.py` implements everything. It is **registry-driven**:

- Column projection comes from `get_dataset_config()` — no hardcoded `"text"` assumptions. Handles:
  - `quran_qa` → keeps `context/question/answer` (registry sets `text_column: context`)
  - `mc4_ar` → passes config name `ar` via `load_dataset_by_name()`
  - `summarization` → keeps native instruction/output mapping
- Every line gets metadata: `source`, `hf_path`, `license`, `domain`, `type`.
- NO cleaning/normalization — tashkeel preserved verbatim.
- Per-directory `manifest.json` with row/char counts.

# Steps to Execute

1. Smoke the script with tiny caps first (~2 min):
   ```bash
   uv run python -m src.scripts.cache_raw_data --stage both --max-samples 50 --output-root D:/AI/Datasets/baligh_smoke --overwrite
   ```
2. Inspect `D:/AI/Datasets/baligh_smoke/cpt/general/manifest.json` and one JSONL line per directory — verify Arabic text intact, metadata present.
3. Delete the smoke dir, then run the full cache:
   ```bash
   uv run python -m src.scripts.cache_raw_data --stage both --output-root D:/AI/Datasets/baligh
   ```
4. Large corpora (`arabicweb24` 28B tokens, `mc4_ar` 50B) stream unbounded — expect hours and ~100+ GB disk. For a bounded cache instead:
   ```bash
   uv run python -m src.scripts.cache_raw_data --stage both --max-samples 2000000 --output-root D:/AI/Datasets/baligh
   ```
5. Record final manifest totals in `docs/data/raw-cache.md`.

# Verification

- `D:\AI\Datasets\baligh\cpt\general\manifest.json` lists all 5 general CPT sets.
- `...\cpt\islamic\manifest.json` lists all 3 Islamic sets.
- `...\sft\raw\manifest.json` lists all 4 SFT sets.
- Spot-check a Quran line: tashkeel must be present.

# Constraints

- NEVER store any of this inside Git (already excluded — it lives on D:).
- Do not clean or normalize here; that is `prepare_data.py`'s job downstream.
- If a single dataset fails, the script logs and continues — report failures, don't abort the batch.
