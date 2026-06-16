# constants.py — Complete Line-by-Line Explanation

**File**: `src/baligh/constants.py` (91 lines)
**Purpose**: Centralizes ALL hardcoded values — filesystem paths, model names, default hyperparameters, dataset ratios, and dataset lists. Single source of truth for magic numbers.

---

## Imports (Lines 1-3)

```python
"""Constants and paths for Baligh-1.5B v0."""
```
Line 1: Module docstring.

```python
from pathlib import Path
```
Line 3: Import Path for cross-platform filesystem path handling.

---

## Project Root (Lines 5-6)

```python
# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
```
Lines 5-6: This is the most important line in the file. It discovers the project root by navigating up from the current file:

```
__file__ = src/baligh/constants.py
.parent   = src/baligh/
.parent   = src/
.parent   = Baligh/          (project root)
.parent   = Projects/        (parent directory)
```

Wait, that's 4 levels up from `constants.py`:
1. `constants.py` → `baligh/` (parent)
2. `baligh/` → `src/` (parent)
3. `src/` → `Baligh/` (parent)
4. `Baligh/` → `Projects/` (parent)

Actually, the correct chain is:
1. `constants.py` is in `src/baligh/`
2. `.parent` = `src/baligh/`
3. `.parent.parent` = `src/`
4. `.parent.parent.parent` = `Baligh/` (project root)
5. `.parent.parent.parent.parent` = parent of project root

This seems like it goes one level too far. The actual project root should be `Path(__file__).parent.parent.parent` (3 levels up from `constants.py`). This is likely a bug in the code — it should be 3 `.parent` calls, not 4. However, the code works because all paths are relative to PROJECT_ROOT, and the extra parent just points to a higher directory.

**Important note**: This is a potential issue. If you're debugging path problems, check whether PROJECT_ROOT actually points to your project directory.

---

## Data Paths (Lines 8-12)

```python
# Data paths
DATA_DIR = PROJECT_ROOT / "data"
```
Line 9: Base directory for all data. The `/` operator joins path segments: `PROJECT_ROOT / "data"` = `<root>/data/`.

```python
RAW_DATA_DIR = DATA_DIR / "raw"
```
Line 10: Where downloaded (unprocessed) datasets are stored. After `prepare_data.py` downloads from Hugging Face.

```python
CLEAN_DATA_DIR = DATA_DIR / "clean"
```
Line 11: Where cleaned datasets go (after HTML removal, normalization, etc.).

```python
TRAIN_READY_DIR = DATA_DIR / "train_ready"
```
Line 12: Final tokenized/formatted data ready for training. This is what `run_cpt.py` and `run_sft.py` read.

---

## Config Paths (Lines 14-19)

```python
# Config paths
CONFIG_DIR = PROJECT_ROOT / "configs"
```
Line 15: Root directory for all YAML configuration files.

```python
BASE_CONFIG_DIR = CONFIG_DIR / "base"
```
Line 16: Base model and hardware configs. Contains `model.yaml`, `tokenizer.yaml`, `hardware.yaml`.

```python
CPT_CONFIG_DIR = CONFIG_DIR / "cpt"
```
Line 17: CPT-specific configs. Contains `cpt-stage1.yaml`, `cpt-stage2.yaml`, `cpt-islamic.yaml`.

```python
SFT_CONFIG_DIR = CONFIG_DIR / "sft"
```
Line 18: SFT-specific configs. Contains `sft-stage1.yaml`, `sft-stage2.yaml`.

```python
EVAL_CONFIG_DIR = CONFIG_DIR / "eval"
```
Line 19: Evaluation configs. Contains `eval-config.yaml`.

---

## Training Paths (Lines 21-25)

```python
# Training paths
TRAINING_DIR = PROJECT_ROOT / "training"
```
Line 22: Root for all training outputs (checkpoints, logs, final models).

```python
CPT_OUTPUT_DIR = TRAINING_DIR / "cpt"
```
Line 23: Where CPT checkpoints and final model are saved.

```python
SFT_OUTPUT_DIR = TRAINING_DIR / "sft"
```
Line 24: Where SFT checkpoints and final model are saved.

```python
LOGS_DIR = TRAINING_DIR / "logs"
```
Line 25: Where log files (JSON/text) are written.

---

## Evaluation Paths (Lines 27-30)

```python
# Evaluation paths
EVAL_DIR = PROJECT_ROOT / "eval"
```
Line 28: Root for evaluation outputs.

```python
EVAL_RESULTS_DIR = EVAL_DIR / "results"
```
Line 29: Raw evaluation results (predictions, scores).

```python
EVAL_REPORTS_DIR = EVAL_DIR / "reports"
```
Line 30: Formatted evaluation reports (JSON, Markdown).

---

## Release Paths (Lines 32-36)

```python
# Release paths
RELEASE_DIR = PROJECT_ROOT / "release"
```
Line 33: Root for release artifacts.

```python
BASE_RELEASE_DIR = RELEASE_DIR / "baligh-1.5b-v0-base"
```
Line 34: CPT-adapted model (before SFT).

```python
INSTRUCT_RELEASE_DIR = RELEASE_DIR / "baligh-1.5b-v0-instruct"
```
Line 35: Instruction-tuned model (after SFT).

```python
EVAL_RELEASE_DIR = RELEASE_DIR / "baligh-1.5b-v0-eval"
```
Line 36: Evaluation results release.

---

## Model Constants (Lines 38-42)

```python
# Model constants
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
```
Line 39: Hugging Face model ID for the base model. This is the full-precision version.

```python
BASE_MODEL_UNSLOTH = "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit"
```
Line 40: Unsloth-optimized version. Pre-quantized to 4-bit with BitsAndBytes. Faster loading for QLoRA training.

```python
MAX_SEQ_LENGTH = 2048  # Will extend to 32768 later
```
Line 41: Maximum sequence length for training. The model supports 32,768 tokens but we train at 2,048 to save memory. The comment indicates future extension.

```python
VOCAB_SIZE = 151936
```
Line 42: Vocabulary size of Qwen2.5. This is the number of unique tokens the tokenizer can represent. Includes Arabic, English, code, and special tokens.

---

## Training Constants (Lines 44-50)

```python
# Training constants
DEFAULT_SEED = 42
```
Line 45: Random seed for reproducibility. Same seed = same results every run.

```python
DEFAULT_BATCH_SIZE = 2
```
Line 46: Batch size per GPU. With gradient accumulation of 4, effective batch = 8.

```python
DEFAULT_GRAD_ACCUM = 4
```
Line 47: Gradient accumulation steps. Simulates larger batch without more memory.

```python
DEFAULT_LR = 2e-4
```
Line 48: Learning rate. 2e-4 = 0.0002. Relatively high for fine-tuning but appropriate for CPT.

```python
DEFAULT_WARMUP_STEPS = 1000
```
Line 49: Linear warmup from 0 to learning_rate over first 1000 steps.

```python
DEFAULT_WEIGHT_DECAY = 0.01
```
Line 50: L2 regularization strength. Prevents weights from growing too large.

---

## LoRA Constants (Lines 52-59)

```python
# LoRA constants
DEFAULT_LORA_R = 16
```
Line 53: LoRA rank. Higher = more expressive but more parameters. 16 is a good balance.

```python
DEFAULT_LORA_ALPHA = 16
```
Line 54: LoRA scaling factor. alpha/r = 1.0 means no scaling.

```python
DEFAULT_LORA_DROPOUT = 0.0
```
Line 55: LoRA dropout. 0.0 = no dropout (Unsloth optimization).

```python
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```
Lines 56-59: Which layers get LoRA adapters. These are ALL major linear layers in the transformer:
- Attention: q_proj (Query), k_proj (Key), v_proj (Value), o_proj (Output)
- MLP: gate_proj, up_proj, down_proj

---

## Dataset Constants (Lines 61-82)

```python
# Dataset constants
CPT_MIX_RATIOS = {
    "arabicweb24": 0.70,
    "arabictext_large": 0.20,
    "arabic_pile": 0.10,
}
```
Lines 62-66: CPT dataset mixing ratios. ArabicWeb24 dominates at 70% because it's the largest and most diverse Arabic web corpus.

```python
SFT_MIX_RATIOS = {
    "cidar": 0.40,
    "evol_instruct_arabic": 0.35,
    "gazelle": 0.10,
    "summarization": 0.10,
    "islamic_qa": 0.05,
}
```
Lines 68-74: SFT dataset mixing ratios. CIDAR is the primary Arabic instruction dataset (40%). Islamic QA is small (5%) to add domain knowledge without overwhelming general instruction-following.

```python
# Islamic corpus datasets (separate cycle)
ISLAMIC_DATASETS = [
    "hadith_datasets",
    "quran_qa",
    "quran_md",
    "arabic_islamic_texts",
]
```
Lines 77-82: Datasets for the Islamic training cycle. This runs as a separate pass after the main CPT.

---

## Evaluation Constants (Lines 84-91)

```python
# Evaluation datasets
EVAL_DATASETS = [
    "mmlu_arabic",
    "cidar_eval",
    "cidar_mcq",
    "mr_tydi_arabic",
    "islamic_qa_custom",
]
```
Lines 85-91: Datasets used for evaluation. Each tests a different capability:
- `mmlu_arabic`: General knowledge (multiple choice)
- `cidar_eval`: Arabic instruction-following
- `cidar_mcq`: Arabic multiple choice
- `mr_tydi_arabic`: Retrieval-augmented QA
- `islamic_qa_custom`: Islamic domain knowledge

---

## Relationship to config.py

`constants.py` provides hardcoded defaults that mirror `config.py` dataclass defaults. The difference:
- `constants.py`: Simple module-level constants. No validation, no env var support. Used in scripts and tests.
- `config.py`: Typed dataclasses with validation, env var loading, and immutable instances. Used in training code.

The constants are useful when you need a quick reference value without importing the full config system.

---

## Design Decisions

1. **Single source of truth**: All magic numbers live here. If you change the learning rate, change it in both `config.py` AND `constants.py`.
2. **No dependencies on other baligh modules**: `constants.py` only imports `pathlib`. This prevents circular imports.
3. **ALL_CAPS naming**: Follows Python convention for module-level constants.
4. **Path objects, not strings**: Using `Path` ensures cross-platform compatibility.
