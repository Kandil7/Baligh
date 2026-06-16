# constants.py - Constants and Paths

**Path**: `src/baligh/constants.py` (91 lines)

## Purpose

Centralizes all hardcoded values: filesystem paths, model names, default hyperparameters, dataset ratios, and dataset lists. Single source of truth for magic numbers.

---

## Path Constants (lines 6-36)

All paths are derived from `PROJECT_ROOT` using `Path(__file__).parent.parent.parent.parent` (traverses from `src/baligh/` to project root).

**Data paths**: `DATA_DIR/data/`, `RAW_DATA_DIR/data/raw/`, `CLEAN_DATA_DIR/data/clean/`, `TRAIN_READY_DIR/data/train_ready/`

**Config paths**: `CONFIG_DIR/configs/`, with subdirectories `base/`, `cpt/`, `sft/`, `eval/`

**Training paths**: `TRAINING_DIR/training/`, `CPT_OUTPUT_DIR/training/cpt/`, `SFT_OUTPUT_DIR/training/sft/`, `LOGS_DIR/training/logs/`

**Evaluation paths**: `EVAL_DIR/eval/`, `EVAL_RESULTS_DIR/eval/results/`, `EVAL_REPORTS_DIR/eval/reports/`

**Release paths**: `RELEASE_DIR/release/`, with subdirectories for base, instruct, and eval releases.

---

## Model Constants (lines 38-42)

- `BASE_MODEL_NAME` = "Qwen/Qwen2.5-1.5B"
- `BASE_MODEL_UNSLOTH` = "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit"
- `MAX_SEQ_LENGTH` = 2048 (noted as extendable to 32768)
- `VOCAB_SIZE` = 151,936

---

## Training Constants (lines 44-49)

Default hyperparameters: seed=42, batch_size=2, grad_accum=4, lr=2e-4, warmup=1000, weight_decay=0.01.

---

## LoRA Constants (lines 52-59)

Default LoRA parameters: r=16, alpha=16, dropout=0.0, target_modules (7 modules).

---

## Dataset Ratios (lines 62-82)

CPT_MIX_RATIOS: arabicweb24=0.70, arabictext_large=0.20, arabic_pile=0.10

SFT_MIX_RATIOS: cidar=0.40, evol_instruct_arabic=0.35, gazelle=0.10, summarization=0.10, islamic_qa=0.05

ISLAMIC_DATASETS: hadith_datasets, quran_qa, quran_md, arabic_islamic_texts

EVAL_DATASETS: mmlu_arabic, cidar_eval, cidar_mcq, mr_tydi_arabic, islamic_qa_custom

---

## Relationship to config.py

`constants.py` provides hardcoded defaults that mirror `config.py` dataclass defaults. The constants are used in places where importing the full config is unnecessary (e.g., test files, simple scripts). The config classes are the authoritative source for runtime values.
