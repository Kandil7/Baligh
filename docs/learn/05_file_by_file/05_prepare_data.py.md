# prepare_data.py - Data Preparation Script

**Path**: `src/scripts/prepare_data.py` (376 lines)

## Purpose

CLI entry point that orchestrates the entire data pipeline: download, clean, deduplicate, validate, mix, format, split, and save Hugging Face datasets for CPT and SFT training.

---

## DataPrepConfig (lines 36-48)

Dataclass holding pipeline configuration: stage, output_dir, clean, deduplicate, streaming, num_proc, seed, max_samples, push_to_hf, hf_repo, hf_token.

---

## DataPreparator (lines 49-240)

Main orchestrator class. Methods:

- `prepare_cpt()` (line 59): Loads CPT datasets, cleans, deduplicates, validates, mixes by ratio, formats with tokenizer, splits train/eval.
- `prepare_sft()` (line 109): Same pipeline for SFT datasets with SFT-specific cleaning.
- `prepare_eval()` (line 151): Loads eval datasets without cleaning/mixing.
- `_clean_dataset()` (line 161): Applies CleaningPipeline to text column.
- `_clean_sft_dataset()` (line 175): Applies cleaning to instruction/input/output columns.
- `_split_dataset()` (line 190): Splits into train/eval (99/1 for CPT, 98/2 for SFT).
- `save()` (line 210): Saves to disk with metadata.json.
- `push_to_hf()` (line 231): Pushes to Hugging Face Hub.

---

## CLI Arguments (lines 241-332)

| Argument | Default | Purpose |
|----------|---------|---------|
| --stage | "both" | cpt, sft, both, eval |
| --output-dir | "data/train_ready" | Output location |
| --clean | False | Apply cleaning pipeline |
| --deduplicate | False | MinHash LSH dedup |
| --streaming | True | Stream large datasets |
| --num-proc | 8 | Parallel workers |
| --max-samples | None | Limit for testing |
| --push-to-hf | False | Upload to Hub |
| --hf-repo | "Kandil7/Baligh-1.5B-v0-data" | Hub repo ID |
