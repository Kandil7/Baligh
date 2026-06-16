# mixer.py - Dataset Mixing

**Path**: `src/baligh/data/mixer.py` (44 lines)

## Purpose

Interleaves multiple datasets by configured probability ratios using HF `interleave_datasets()`.

---

## DatasetMixer (lines 9-28)

Takes a `mix_config` dict (name to ratio), seed, and stopping_strategy. The `mix()` method:
1. Filters datasets to only those present in the mix_config
2. Normalizes ratios to probabilities (divides by total)
3. Calls `interleave_datasets()` with the filtered datasets and probabilities

Stopping strategy `first_exhausted` stops when the smallest dataset runs out.

---

## Convenience Functions

- `get_cpt_mixer(seed)` - Creates mixer with CPTConfig ratios
- `get_sft_mixer(seed)` - Creates mixer with SFTConfig ratios
- `mix_cpt_datasets(datasets, seed)` - One-shot CPT mixing
- `mix_sft_datasets(datasets, seed)` - One-shot SFT mixing
