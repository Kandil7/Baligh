# datasets.py - Dataset Registry

**Path**: `src/baligh/data/datasets.py` (75 lines)

## Purpose

Central registry of all datasets used in the Baligh project. Maps human-readable names to Hugging Face paths, splits, column names, licenses, and metadata.

---

## The DATASETS Dictionary (lines 7-53)

A single dictionary with 13 entries. Each entry contains:
- `path`: Hugging Face dataset ID
- `split`: Dataset split to load (train or test)
- `streaming`: Whether to use streaming mode
- `text_column` or `instruction_column`/`output_column`: Column name mappings
- `license`: License type
- `tokens`: Estimated token count
- `domain`: Domain category (web, general, mixed, islamic, instruction, summarization, eval)
- `type`: Pipeline type (cpt, cpt_islamic, sft, eval)

---

## Helper Functions

- `get_dataset_config(name)` (line 55): Returns config dict for a single dataset
- `list_datasets(dataset_type=None)` (line 60): Filters by type
- `get_cpt_datasets()` (line 65): Returns type='cpt' datasets
- `get_sft_datasets()` (line 68): Returns type='sft' datasets
- `get_eval_datasets()` (line 71): Returns type='eval' datasets
- `get_islamic_datasets()` (line 74): Returns type='cpt_islamic' datasets
