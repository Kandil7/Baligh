# datasets.py — Complete Line-by-Line Explanation

**File**: `src/baligh/data/datasets.py` (75 lines)
**Purpose**: Central registry of ALL datasets used in the Baligh project. Maps human-readable names to Hugging Face paths, splits, column names, licenses, and metadata.

---

## Imports (Lines 1-5)

```python
"""Dataset registry for Baligh-1.5B v0."""
```
Line 1: Module docstring.

```python
from baligh.utils.logging import get_logger
```
Line 3: Import the project's logging utility. `get_logger(__name__)` creates a logger bound to this module's name.

```python
logger = get_logger(__name__)
```
Line 5: Create the logger instance. All log messages from this module will be prefixed with the module name.

---

## The DATASETS Dictionary (Lines 7-53)

This is the heart of the file. A single dictionary mapping dataset names to their configuration.

### CPT Datasets (Lines 8-22)

```python
DATASETS = {
    'arabicweb24': {
        'path': 'lightonai/ArabicWeb24', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 28000000000, 'domain': 'web', 'type': 'cpt',
    },
```
Lines 8-10: ArabicWeb24 dataset.
- `path`: Hugging Face dataset ID. `lightonai/ArabicWeb24` is the repo on HF Hub.
- `split`: Which split to load. "train" = training data.
- `streaming`: True = load lazily from HF Hub (don't download 28B tokens!). False = download full dataset.
- `text_column`: The column name containing the text. HF datasets can have multiple columns; this tells us which one has the text.
- `license`: "Open" means permissive license.
- `tokens`: Estimated token count. 28 billion tokens. This is used for planning and logging.
- `domain`: "web" = web-crawled text.
- `type`: "cpt" = used in Continued Pretraining.

```python
    'arabictext_large': {
        'path': 'Jr23xd23/ArabicText-Large', 'split': 'train', 'streaming': False, 'text_column': 'text', 'license': 'Apache-2.0', 'tokens': 1000000000, 'domain': 'general', 'type': 'cpt',
    },
```
Lines 11-13: ArabicText-Large.
- `streaming: False`: Download the full dataset (only 1B tokens, fits in memory).
- `license`: Apache-2.0 (permissive, requires attribution).
- `tokens`: 1 billion tokens.
- `domain`: "general" = general Arabic text (books, articles, etc.).

```python
    'arabic_pile': {
        'path': 'premio-ai/TheArabicPile_Dialects', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 5000000000, 'domain': 'mixed', 'type': 'cpt',
    },
```
Lines 14-16: The Arabic Pile (Dialects).
- `streaming: True`: 5B tokens, stream to save memory.
- `domain`: "mixed" = includes various Arabic dialects (Egyptian, Levantine, Gulf, etc.).

### Additional CPT Datasets (Lines 17-22)

```python
    'oscar_ar': {
        'path': 'oscar-corpus/oscar_ar', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 10000000000, 'domain': 'web', 'type': 'cpt',
    },
```
Lines 17-19: OSCAR Arabic. 10B tokens of web-crawled Arabic text. Not in the default mix but available for expansion.

```python
    'mc4_ar': {
        'path': 'google/mc4', 'name': 'ar', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 50000000000, 'domain': 'web', 'type': 'cpt',
    },
```
Lines 20-22: mC4 Arabic. 50B tokens! The largest Arabic web corpus. Note the `'name': 'ar'` field — mC4 is a multilingual dataset, so we specify the Arabic subset.

### Islamic CPT Datasets (Lines 23-31)

```python
    'hadith_datasets': {
        'path': 'meeAtif/hadith_datasets', 'split': 'train', 'streaming': False, 'text_column': 'text', 'license': 'Open', 'domain': 'islamic', 'type': 'cpt_islamic',
    },
```
Lines 23-25: Hadith collections. Reports of the Prophet's sayings and actions. `type: 'cpt_islamic'` = used in the Islamic training cycle.

```python
    'quran_qa': {
        'path': 'nazimali/quran-question-answer-context', 'split': 'train', 'streaming': False, 'text_column': 'context', 'license': 'CC', 'domain': 'islamic', 'type': 'cpt_islamic',
    },
```
Lines 26-28: Quran Q&A. Note `text_column: 'context'` — the text column is named "context" in this dataset, not "text". License is CC (Creative Commons).

```python
    'quran_md': {
        'path': 'Buraaq/quran-audio-text-dataset', 'split': 'train', 'streaming': False, 'text_column': 'text', 'license': 'Open', 'domain': 'islamic', 'type': 'cpt_islamic',
    },
```
Lines 29-31: Quran text (from audio transcription dataset). Raw Quranic text for language modeling.

### SFT Datasets (Lines 32-43)

```python
    'cidar': {
        'path': 'arbml/CIDAR', 'split': 'train', 'streaming': False, 'instruction_column': 'instruction', 'input_column': 'input', 'output_column': 'output', 'license': 'Apache-2.0', 'domain': 'instruction', 'type': 'sft',
    },
```
Lines 32-34: CIDAR — the primary Arabic instruction dataset.
- `instruction_column`: Column name for the instruction.
- `input_column`: Column name for optional input/context.
- `output_column`: Column name for the expected response.
- These column names are different from CPT datasets because SFT data has structured instruction-response pairs.

```python
    'evol_instruct_arabic': {
        'path': 'FreedomIntelligence/evol-instruct-arabic', 'split': 'train', 'streaming': False, 'instruction_column': 'instruction', 'input_column': 'input', 'output_column': 'output', 'license': 'Open', 'domain': 'instruction', 'type': 'sft',
    },
```
Lines 35-37: Evolved Instructions Arabic. Instructions that have been evolved (made more complex) through a self-instruct pipeline.

```python
    'gazelle': {
        'path': 'Gazelle/arabic-writing', 'split': 'train', 'streaming': False, 'instruction_column': 'instruction', 'input_column': 'input', 'output_column': 'output', 'license': 'Open', 'domain': 'instruction', 'type': 'sft',
    },
```
Lines 38-40: Gazelle Arabic Writing. Instruction data focused on Arabic writing tasks.

```python
    'summarization': {
        'path': 'BounharAbdelaziz/arabic-msa-summarization', 'split': 'train', 'streaming': False, 'instruction_column': 'text', 'output_column': 'summary', 'license': 'Open', 'domain': 'summarization', 'type': 'sft',
    },
```
Lines 41-43: Arabic MSA Summarization. Note the different column names: `instruction_column: 'text'` and `output_column: 'summary'`. This dataset has text/summary pairs, not instruction/output pairs.

### Eval Datasets (Lines 44-52)

```python
    'mmlu_arabic': {
        'path': 'FreedomIntelligence/MMLU_Arabic', 'split': 'test', 'streaming': False, 'license': 'Open', 'domain': 'eval', 'type': 'eval',
    },
```
Lines 44-46: MMLU Arabic. Multiple-choice knowledge benchmark. Uses the "test" split (not "train").

```python
    'cidar_eval': {
        'path': 'arbml/CIDAR', 'split': 'test', 'streaming': False, 'license': 'Apache-2.0', 'domain': 'eval', 'type': 'eval',
    },
```
Lines 47-49: CIDAR evaluation. Same dataset as SFT but using the "test" split for evaluation.

```python
    'mr_tydi_arabic': {
        'path': 'castorini/mr-tydi', 'name': 'arabic', 'split': 'test', 'streaming': False, 'license': 'Open', 'domain': 'eval', 'type': 'eval',
    },
```
Lines 50-52: Mr. TyDi Arabic. Retrieval-augmented QA benchmark. Note `'name': 'arabic'` — this is a multilingual dataset.

---

## Helper Functions (Lines 55-75)

```python
def get_dataset_config(name):
    if name not in DATASETS:
        raise ValueError('Unknown dataset: %s' % name)
    return DATASETS[name]
```
Lines 55-58: Get configuration for a single dataset by name. Raises ValueError if the name doesn't exist.

```python
def list_datasets(dataset_type=None):
    if dataset_type:
        return {k: v for k, v in DATASETS.items() if v.get('type') == dataset_type}
    return DATASETS
```
Lines 60-63: List datasets, optionally filtered by type. Uses a dict comprehension to filter.

```python
def get_cpt_datasets():
    return list_datasets('cpt')
```
Lines 65-66: Convenience function for CPT datasets.

```python
def get_sft_datasets():
    return list_datasets('sft')
```
Lines 68-69: Convenience function for SFT datasets.

```python
def get_eval_datasets():
    return list_datasets('eval')
```
Lines 71-72: Convenience function for eval datasets.

```python
def get_islamic_datasets():
    return list_datasets('cpt_islamic')
```
Lines 74-75: Convenience function for Islamic CPT datasets.

---

## Design Decisions

1. **Single dictionary**: All datasets in one place. Adding a new dataset = one dict entry.
2. **Type filtering**: Can filter by 'cpt', 'sft', 'eval', 'cpt_islamic' for different pipeline stages.
3. **Column name mapping**: Different datasets have different column names. The registry maps them to standard names.
4. **Token estimates**: Helps with planning (how much disk space, how long to train).
5. **License tracking**: Important for legal compliance when releasing the model.
