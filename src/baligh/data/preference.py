"""Preference data handling for DPO/ORPO training.

Canonical format (JSONL or HuggingFace dataset):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Accepted alias:
    {"instruction": "...", "chosen": "...", "rejected": "..."}

``normalize_preference_schema`` renames the alias to the canonical column so
TRL's DPOTrainer always sees prompt/chosen/rejected. Tokenization happens
inside TRL (driven by DPOConfig.max_length / max_prompt_length) — this module
deliberately does NOT pre-tokenize.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

PREFERENCE_COLUMNS = ("prompt", "chosen", "rejected")
PROMPT_ALIAS = "instruction"


def validate_preference_example(example: dict) -> bool:
    """Check that a preference example has all required fields non-empty."""
    has_prompt = bool(example.get("prompt") or example.get(PROMPT_ALIAS))
    return has_prompt and bool(example.get("chosen")) and bool(example.get("rejected"))


def normalize_preference_schema(dataset: Any) -> Any:
    """Ensure the dataset exposes canonical prompt/chosen/rejected columns.

    Renames ``instruction`` -> ``prompt`` when present. Raises ValueError
    when neither schema can be satisfied, so a mis-formatted file fails
    loudly before training instead of mid-run.
    """
    columns = set(dataset.column_names or [])
    if {"prompt", "chosen", "rejected"} <= columns:
        return dataset
    if PROMPT_ALIAS in columns and {"chosen", "rejected"} <= columns:
        logger.info(f"Preference data uses '{PROMPT_ALIAS}' alias; renaming to 'prompt'")
        return dataset.rename_column(PROMPT_ALIAS, "prompt")
    raise ValueError(
        f"Preference dataset must have {PREFERENCE_COLUMNS} columns "
        f"(or '{PROMPT_ALIAS}' as prompt alias). Got: {sorted(columns)}"
    )


def load_preference_data(
    data_path: str | Path,
    split: str = "train",
    streaming: bool = False,
) -> Dataset:
    """Load preference data from a local file or HuggingFace dataset.

    Supports JSONL, CSV, Parquet, an Arrow directory saved via
    ``save_to_disk``, or a HF dataset repo ID.
    """
    data_path_str = str(data_path)
    path = Path(data_path_str)

    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            ds = load_dataset("json", data_files=data_path_str, split=split, streaming=streaming)
        elif suffix == ".csv":
            ds = load_dataset("csv", data_files=data_path_str, split=split, streaming=streaming)
        elif suffix == ".parquet":
            ds = load_dataset("parquet", data_files=data_path_str, split=split, streaming=streaming)
        else:
            loaded = load_from_disk(data_path_str)
            ds = loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
    else:
        ds = load_dataset(data_path_str, split=split, streaming=streaming)

    logger.info(f"Loaded preference data: {len(ds)} examples from {data_path_str}")
    return ds


def get_preference_stats(dataset: Any) -> dict[str, Any]:
    """Compute basic word-length statistics for a preference dataset."""
    chosen_lengths = []
    rejected_lengths = []
    for example in dataset:
        chosen_lengths.append(len(str(example.get("chosen") or "").split()))
        rejected_lengths.append(len(str(example.get("rejected") or "").split()))

    n = len(chosen_lengths)
    return {
        "num_examples": len(dataset),
        "chosen_mean_words": sum(chosen_lengths) / n if n else 0,
        "rejected_mean_words": sum(rejected_lengths) / n if n else 0,
        "chosen_max_words": max(chosen_lengths) if chosen_lengths else 0,
        "rejected_max_words": max(rejected_lengths) if rejected_lengths else 0,
    }
