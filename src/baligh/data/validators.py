"""Data validation for Baligh-1.5B v0."""

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_CPT_COLUMNS = ["text"]
REQUIRED_SFT_COLUMNS = ["instruction", "output"]
OPTIONAL_SFT_COLUMNS = ["input"]


def validate_cpt_example(example):
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


def validate_sft_example(example):
    for col in REQUIRED_SFT_COLUMNS:
        if not example.get(col):
            return False, "Missing %s field" % col
        if not isinstance(example[col], str):
            return False, "%s field must be string" % col
    if len(example["instruction"]) < 5:
        return False, "Instruction too short"
    if len(example["output"]) < 5:
        return False, "Output too short"
    return True, ""


def validate_dataset(dataset, dataset_type="cpt"):
    if dataset_type == "cpt":
        validator = validate_cpt_example
    else:
        validator = validate_sft_example
    valid_count = 0
    invalid_count = 0
    errors = {}
    for example in dataset:
        valid, error = validator(example)
        if valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors[error] = errors.get(error, 0) + 1
    logger.info("Validation: %d valid, %d invalid" % (valid_count, invalid_count))
    if errors:
        logger.warning("Validation errors: %s" % errors)
    return valid_count, invalid_count, errors


def check_dataset_schema(dataset, required_columns):
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError("Missing columns: %s" % missing)
    return True


def get_dataset_stats(dataset, text_column="text"):
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
