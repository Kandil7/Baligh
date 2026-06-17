# loader.py — Complete Line-by-Line Explanation

**File**: `src/baligh/data/loader.py` (49 lines)
**Purpose**: Loads Hugging Face datasets by name using a local registry. Provides a unified interface for loading any dataset in the project.

---

## Imports (Lines 1-7)

```python
"""Dataset loading for Baligh-1.5B v0."""
```
Line 1: Module docstring.

```python
from datasets import load_dataset, concatenate_datasets, interleave_datasets
```
Line 3: Import HF datasets functions:
- `load_dataset`: Downloads and loads a dataset from HF Hub
- `concatenate_datasets`: Joins datasets end-to-end (all rows from dataset A, then all from B)
- `interleave_datasets`: Alternates between datasets by probability (used in mixer.py)

```python
from baligh.config import get_cpt_config, get_sft_config, get_base_config
```
Line 4: Import config factories. These provide the configuration values needed for loading.

```python
from baligh.utils.logging import get_logger
```
Line 5: Import the project's logging utility.

```python
logger = get_logger(__name__)
```
Line 7: Create a logger bound to this module.

---

## DATASET_REGISTRY (Lines 9-38)

```python
# Dataset registry with metadata
DATASET_REGISTRY = {
```
Line 10: A SECOND registry (in addition to `datasets.py`). This one is specifically for the `load_dataset_by_name()` function and includes only CPT datasets with full loading metadata.

```python
    "arabicweb24": {
        "path": "lightonai/ArabicWeb24",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens_estimate": 28000000000,
        "domain": "web",
    },
```
Lines 11-19: ArabicWeb24 configuration for loading.
- `path`: The HF dataset ID to pass to `load_dataset()`
- `split`: Which split to load
- `streaming`: Whether to stream (lazy load) or download fully
- `text_column`: Which column contains the text
- `license`: For reference
- `tokens_estimate`: Estimated token count for planning
- `domain`: For reference

```python
    "arabictext_large": {
        "path": "Jr23xd23/ArabicText-Large",
        "split": "train",
        "streaming": False,
        "text_column": "text",
        "license": "Apache-2.0",
        "tokens_estimate": 1000000000,
        "domain": "general",
    },
```
Lines 20-28: ArabicText-Large. `streaming: False` because it's small enough to download.

```python
    "arabic_pile": {
        "path": "premio-ai/TheArabicPile_Dialects",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens_estimate": 5000000000,
        "domain": "mixed",
    },
```
Lines 29-37: The Arabic Pile. Streaming enabled due to size.

---

## load_dataset_by_name (Lines 40-49)

```python
def load_dataset_by_name(name: str, split: str = None, streaming: bool = None, **kwargs):
```
Line 40: The main loading function.
- `name`: Dataset name (must be in DATASET_REGISTRY)
- `split`: Override the default split (None = use registry default)
- `streaming`: Override the default streaming setting
- `**kwargs`: Any additional arguments passed to HF `load_dataset()`

```python
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
```
Lines 41-42: Validate that the dataset name exists. If not, raise an error with a helpful message.

```python
    config = DATASET_REGISTRY[name].copy()
```
Line 43: Get the dataset config and make a copy. We copy because we're about to modify it with `.pop()`.

```python
    split = split or config.pop("split", "train")
```
Line 44: Use the provided split, or fall back to the registry's default. `config.pop("split", "train")` removes "split" from the config dict (so it doesn't get passed to `load_dataset()` twice) and returns its value.

```python
    streaming = streaming if streaming is not None else config.pop("streaming", False)
```
Line 45: Same logic for streaming. Use the provided value, or fall back to the registry default.

```python
    path = config.pop("path")
```
Line 46: Extract the HF dataset path and remove it from the config dict.

```python
    name_param = config.pop("name", None)
```
Line 47: Extract the optional dataset name (for multilingual datasets like mC4). Remove from config dict.

```python
    logger.info(f"Loading dataset: {name} (split={split}, streaming={streaming})")
```
Line 48: Log what we're about to load.

```python
    return load_dataset(path, name=name_param, split=split, streaming=streaming, **config, **kwargs)
```
Line 49: Call HF `load_dataset()` with all the extracted parameters. The remaining `config` dict contains extra fields (like `text_column`, `license`, etc.) that get passed as kwargs to `load_dataset()`. Note: `load_dataset()` will ignore unknown kwargs, so extra fields are safe.

---

## Design Decisions

1. **Copy before pop**: `config = DATASET_REGISTRY[name].copy()` prevents modifying the original registry.
2. **Pop for extraction**: Using `.pop()` both extracts values AND removes them from the dict, preventing duplicate kwargs.
3. **Override support**: The function parameters override registry defaults, allowing flexible loading.
4. **Streaming by default for large datasets**: ArabicWeb24 and Arabic Pile stream to avoid downloading 30+ GB.

---

## Usage Example

```python
# Load ArabicWeb24 (streaming)
ds = load_dataset_by_name("arabicweb24")

# Load ArabicText-Large (non-streaming)
ds = load_dataset_by_name("arabictext_large")

# Override split
ds = load_dataset_by_name("mmlu_arabic", split="test")

# Override streaming
ds = load_dataset_by_name("arabicweb24", streaming=False)  # Downloads full dataset
```
