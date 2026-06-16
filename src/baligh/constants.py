"""Constants and paths for Baligh-1.5B v0."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
TRAIN_READY_DIR = DATA_DIR / "train_ready"

# Config paths
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG_DIR = CONFIG_DIR / "base"
CPT_CONFIG_DIR = CONFIG_DIR / "cpt"
SFT_CONFIG_DIR = CONFIG_DIR / "sft"
EVAL_CONFIG_DIR = CONFIG_DIR / "eval"

# Training paths
TRAINING_DIR = PROJECT_ROOT / "training"
CPT_OUTPUT_DIR = TRAINING_DIR / "cpt"
SFT_OUTPUT_DIR = TRAINING_DIR / "sft"
LOGS_DIR = TRAINING_DIR / "logs"

# Evaluation paths
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"
EVAL_REPORTS_DIR = EVAL_DIR / "reports"

# Release paths
RELEASE_DIR = PROJECT_ROOT / "release"
BASE_RELEASE_DIR = RELEASE_DIR / "baligh-1.5b-v0-base"
INSTRUCT_RELEASE_DIR = RELEASE_DIR / "baligh-1.5b-v0-instruct"
EVAL_RELEASE_DIR = RELEASE_DIR / "baligh-1.5b-v0-eval"

# Model constants
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BASE_MODEL_UNSLOTH = "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048  # Will extend to 32768 later
VOCAB_SIZE = 151936

# Training constants
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_WEIGHT_DECAY = 0.01

# LoRA constants
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Dataset constants
CPT_MIX_RATIOS = {
    "arabicweb24": 0.70,
    "arabictext_large": 0.20,
    "arabic_pile": 0.10,
}

SFT_MIX_RATIOS = {
    "cidar": 0.40,
    "evol_instruct_arabic": 0.35,
    "gazelle": 0.10,
    "summarization": 0.10,
    "islamic_qa": 0.05,
}

# Islamic corpus datasets (separate cycle)
ISLAMIC_DATASETS = [
    "hadith_datasets",
    "quran_qa",
    "quran_md",
    "arabic_islamic_texts",
]

# Evaluation datasets
EVAL_DATASETS = [
    "mmlu_arabic",
    "cidar_eval",
    "cidar_mcq",
    "mr_tydi_arabic",
    "islamic_qa_custom",
]
