# validators.py - Data Validation

**Path**: `src/baligh/data/validators.py` (74 lines)

## Purpose

Validates dataset schema and example quality before training.

---

## Constants (lines 8-10)

- `REQUIRED_CPT_COLUMNS` = ['text']
- `REQUIRED_SFT_COLUMNS` = ['instruction', 'output']
- `OPTIONAL_SFT_COLUMNS` = ['input']

---

## Validation Functions

### validate_cpt_example (lines 12-22)

Checks: text exists, is string, length between 50 and 100,000 chars.

### validate_sft_example (lines 24-34)

Checks: instruction and output exist, are strings, each >= 5 chars.

### validate_dataset (lines 36-54)

Iterates over dataset, runs appropriate validator, returns (valid_count, invalid_count, errors_dict).

### check_dataset_schema (lines 56-60)

Verifies all required columns exist in dataset.

### get_dataset_stats (lines 62-74)

Returns length statistics (mean, median, min, max, total) for a text column.
