"""Data validation for Baligh-1.7B v0."""

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

from typing import Any

REQUIRED_CPT_COLUMNS: list[str] = ["text"]
REQUIRED_SFT_COLUMNS = ["instruction", "output"]
OPTIONAL_SFT_COLUMNS = ["input"]


def validate_cpt_example(example: dict) -> tuple[bool, str]:
    if not example.get("text"):
        return False, "Missing text field"
    text = example["text"]
    if not isinstance(text, str):
        return False, "Text field must be string"
    if len(text) < 50:
        return False, "Text too short"
    if len(text) > 100000:
        return False, "Text too long"
    return True, ""


def validate_sft_example(example: dict) -> tuple[bool, str]:
    for col in REQUIRED_SFT_COLUMNS:
        if not example.get(col):
            return False, f"Missing {col} field"
        if not isinstance(example[col], str):
            return False, f"{col} field must be string"
    if len(example["instruction"]) < 5:
        return False, "Instruction too short"
    if len(example["output"]) < 5:
        return False, "Output too short"
    return True, ""


def validate_dataset(
    dataset: Any,
    dataset_type: str = "cpt",
    max_samples: int | None = None,
) -> tuple[int, int, dict[str, int]]:
    """Validate examples and return aggregate counts.

    Args:
        dataset: Dataset (map-style or iterable) to validate.
        dataset_type: "cpt" or "sft".
        max_samples: Validate at most this many examples. On large or
            streaming datasets a bounded sample is the sane default ”
            iterating an entire web corpus just to log counts wastes hours
            of bandwidth for decorative numbers. None = validate everything.
    """
    validator = validate_cpt_example if dataset_type == "cpt" else validate_sft_example
    valid_count = 0
    invalid_count = 0
    errors: dict[str, int] = {}
    checked = 0
    for example in dataset:
        if max_samples is not None and checked >= max_samples:
            break
        checked += 1
        valid, error = validator(example)
        if valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors[error] = errors.get(error, 0) + 1
    scope = (
        f" (sampled {checked})"
        if max_samples is not None and checked < _estimate_len(dataset)
        else ""
    )
    logger.info(f"Validation{scope}: {valid_count} valid, {invalid_count} invalid")
    if errors:
        logger.warning(f"Validation errors: {errors}")
    return valid_count, invalid_count, errors


def _estimate_len(dataset: Any) -> int:
    try:
        return len(dataset)
    except TypeError:
        return 0


def check_dataset_schema(dataset: Any, required_columns: list[str]) -> bool:
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True


def get_dataset_stats(dataset: Any, text_column: str = "text") -> dict:
    lengths = [len(ex[text_column]) for ex in dataset if ex.get(text_column)]
    if not lengths:
        return {}
    import numpy as np

    return {
        "count": len(lengths),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
    }
