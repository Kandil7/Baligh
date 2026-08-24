# mixer.py — Complete Line-by-Line Explanation

**File**: `src/baligh/data/mixer.py` (44 lines)
**Purpose**: Mixes multiple datasets by configured probability ratios using HF `interleave_datasets()`. This is how we combine ArabicWeb24 (70%), ArabicText-Large (20%), and Arabic Pile (10%) into a single training stream.

---

## Imports (Lines 1-7)

```python
"""Dataset mixing for Baligh-1.7B v0."""
```
Line 1: Module docstring.

```python
from datasets import Dataset, interleave_datasets
```
Line 3: Import `Dataset` (for type hints) and `interleave_datasets` (the core mixing function).

```python
from baligh.config import get_cpt_config, get_sft_config
```
Line 4: Import config factories to get the default mixing ratios.

```python
from baligh.utils.logging import get_logger
```
Line 5: Import logging utility.

```python
logger = get_logger(__name__)
```
Line 7: Create logger.

---

## DatasetMixer Class (Lines 9-28)

```python
class DatasetMixer:
```
Line 9: The main mixing class. Encapsulates the mixing logic.

```python
    def __init__(self, mix_config, seed=42, stopping_strategy='first_exhausted'):
```
Line 10: Constructor.
- `mix_config`: A dict like `{"arabicweb24": 0.70, "arabictext_large": 0.20}` mapping dataset names to ratios.
- `seed`: Random seed for reproducible mixing.
- `stopping_strategy`: When to stop mixing. `'first_exhausted'` = stop when the smallest dataset runs out.

```python
        self.mix_config = mix_config
        self.seed = seed
        self.stopping_strategy = stopping_strategy
```
Lines 11-13: Store the parameters as instance attributes.

```python
    def mix(self, datasets):
```
Line 15: The mixing method. Takes a dict of datasets (name -> Dataset object).

```python
        ordered_datasets = []
        ordered_probs = []
```
Lines 16-17: Lists to hold the datasets and their probabilities in the same order.

```python
        total = sum(self.mix_config.values())
```
Line 18: Sum all ratios. This is used to normalize ratios to probabilities. For example, if ratios are {0.70, 0.20, 0.10}, total = 1.0, so probabilities stay the same. If ratios were {70, 20, 10}, total = 100, and we'd divide by 100.

```python
for name, ratio in self.mix_config.items():
    if name in datasets:
        ordered_datasets.append(datasets[name])
        ordered_probs.append(ratio / total)
    else:
        logger.warning("Dataset %s not found, skipping" % name)
```
Lines 19-24: Iterate over the mix config. For each dataset:
1. Check if the dataset exists in the provided datasets dict
2. If yes: add it to the ordered lists and normalize the ratio to a probability
3. If no: log a warning and skip it (graceful degradation)

```python
if not ordered_datasets:
    raise ValueError("No valid datasets to mix")
```
Lines 25-26: If no datasets were found, raise an error. We can't mix nothing.

```python
logger.info(
    "Mixing datasets with ratios: %s" % dict(zip([str(d) for d in ordered_datasets], ordered_probs))
)
```
Line 27: Log what we're mixing and at what probabilities.

```python
return interleave_datasets(
    ordered_datasets,
    probabilities=ordered_probs,
    seed=self.seed,
    stopping_strategy=self.stopping_strategy,
)
```
Line 28: Call HF's `interleave_datasets()`:
- `ordered_datasets`: List of Dataset objects to interleave
- `ordered_probs`: Probability of sampling from each dataset
- `seed`: For reproducibility
- `stopping_strategy`: `'first_exhausted'` = stop when the smallest dataset runs out

The result is a single Dataset that alternates between the input datasets according to the probabilities.

---

## Convenience Functions (Lines 30-44)

```python
def get_cpt_mixer(seed=42):
    config = get_cpt_config()
    return DatasetMixer(config.dataset_mix, seed=seed)
```
Lines 30-32: Create a mixer for CPT using the default CPT ratios from config.

```python
def get_sft_mixer(seed=42):
    config = get_sft_config()
    return DatasetMixer(config.dataset_mix, seed=seed)
```
Lines 34-36: Create a mixer for SFT using the default SFT ratios.

```python
def mix_cpt_datasets(datasets, seed=42):
    mixer = get_cpt_mixer(seed)
    return mixer.mix(datasets)
```
Lines 38-40: One-shot CPT mixing. Creates a mixer and immediately mixes.

```python
def mix_sft_datasets(datasets, seed=42):
    mixer = get_sft_mixer(seed)
    return mixer.mix(datasets)
```
Lines 42-44: One-shot SFT mixing.

---

## How interleave_datasets Works

When you call `interleave_datasets([A, B, C], probabilities=[0.7, 0.2, 0.1])`:
1. For each sample, randomly choose which dataset to sample from (70% A, 20% B, 10% C)
2. Take the next sample from that dataset
3. Repeat until one dataset is exhausted (with `first_exhausted` strategy)

This creates a single stream that respects the configured ratios.

---

## Design Decisions

1. **Dict-based config**: Easy to add/remove datasets by editing the config dict.
2. **Graceful degradation**: Missing datasets are skipped with a warning, not an error.
3. **Normalized probabilities**: Ratios don't need to sum to 1.0 — they're normalized automatically.
4. **First exhausted**: Prevents one dataset from dominating indefinitely. When the smallest dataset runs out, training stops.
