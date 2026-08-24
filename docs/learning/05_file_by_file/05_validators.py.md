# validators.py — Complete Line-by-Line Explanation

**File**: `src/baligh/data/validators.py` (74 lines)
**Purpose**: Validates dataset schema and example quality before training. Catches malformed data that would cause training errors.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: Import Dataset type.

Line 5: Import logging.

Line 6: Create logger.

---

## Constants (Lines 8-10)

```python
REQUIRED_CPT_COLUMNS = ["text"]
```
Line 8: CPT datasets must have a 'text' column. This is the raw text for language modeling.

```python
REQUIRED_SFT_COLUMNS = ["instruction", "output"]
```
Line 9: SFT datasets must have 'instruction' and 'output' columns. 'input' is optional.

```python
OPTIONAL_SFT_COLUMNS = ["input"]
```
Line 10: 'input' column is optional for SFT. Some instructions need additional context, some don't.

---

## validate_cpt_example (Lines 12-22)

```python
def validate_cpt_example(example):
```
Line 12: Validate a single CPT example. Returns (is_valid, error_message).

```python
if not example.get("text"):
    return False, "Missing text field"
```
Lines 13-14: Check if 'text' field exists and is not empty. `.get('text')` returns None if missing, and `not None` is True.

```python
text = example["text"]
if not isinstance(text, str):
    return False, "Text field must be string"
```
Lines 15-17: Check that text is actually a string (not a list, dict, or other type).

```python
if len(text) < 50:
    return False, "Text too short"
```
Lines 18-19: Text must be at least 50 characters. Shorter texts are usually noise (HTML fragments, single words, etc.).

```python
if len(text) > 100000:
    return False, "Text too long"
```
Lines 20-21: Text must be under 100,000 characters. Longer texts are likely HTML leakage or data errors.

```python
return True, ""
```
Line 22: All checks passed. Return True with empty error message.

---

## validate_sft_example (Lines 24-34)

```python
def validate_sft_example(example):
```
Line 24: Validate a single SFT example.

```python
for col in REQUIRED_SFT_COLUMNS:
    if not example.get(col):
        return False, "Missing %s field" % col
    if not isinstance(example[col], str):
        return False, "%s field must be string" % col
```
Lines 25-29: Check that required columns (instruction, output) exist and are strings.

```python
if len(example["instruction"]) < 5:
    return False, "Instruction too short"
```
Lines 30-31: Instruction must be at least 5 characters. Very short instructions are usually malformed.

```python
if len(example["output"]) < 5:
    return False, "Output too short"
```
Lines 32-33: Output must be at least 5 characters.

```python
return True, ""
```
Line 34: All checks passed.

---

## validate_dataset (Lines 36-54)

```python
def validate_dataset(dataset, dataset_type='cpt'):
```
Line 36: Validate an entire dataset. Takes a Dataset object and type ('cpt' or 'sft').

```python
if dataset_type == "cpt":
    validator = validate_cpt_example
else:
    validator = validate_sft_example
```
Lines 37-40: Select the appropriate validator based on dataset type.

```python
    valid_count = 0
    invalid_count = 0
    errors = {}
```
Lines 41-43: Initialize counters. `errors` is a dict mapping error messages to their counts.

```python
    for example in dataset:
        valid, error = validator(example)
        if valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors[error] = errors.get(error, 0) + 1
```
Lines 44-50: Iterate over every example in the dataset:
- Run the validator
- Count valid/invalid examples
- Track error types and their frequencies

```python
logger.info("Validation: %d valid, %d invalid" % (valid_count, invalid_count))
```
Line 51: Log the validation results.

```python
if errors:
    logger.warning("Validation errors: %s" % errors)
```
Lines 52-53: Log error details if any were found.

```python
    return valid_count, invalid_count, errors
```
Line 54: Return the validation results.

---

## check_dataset_schema (Lines 56-60)

```python
def check_dataset_schema(dataset, required_columns):
```
Line 56: Check that a dataset has all required columns.

```python
    missing = [col for col in required_columns if col not in dataset.column_names]
```
Line 57: Find columns that are in required_columns but not in the dataset.

```python
if missing:
    raise ValueError("Missing columns: %s" % missing)
```
Lines 58-59: If any required columns are missing, raise an error.

```python
    return True
```
Line 60: All columns present.

---

## get_dataset_stats (Lines 62-74)

```python
def get_dataset_stats(dataset, text_column='text'):
```
Line 62: Compute statistics for a text column in a dataset.

```python
    lengths = [len(ex[text_column]) for ex in dataset if ex.get(text_column)]
```
Line 63: Get the length of each text. Skip examples where the text column is missing.

```python
    if not lengths:
        return {}
```
Lines 64-65: If no valid lengths, return empty dict.

```python
    import numpy as np
```
Line 66: Import numpy inside the function to avoid slow import at module load time.

```python
return {
    "count": len(lengths),
    "mean_length": float(np.mean(lengths)),
    "median_length": float(np.median(lengths)),
    "min_length": min(lengths),
    "max_length": max(lengths),
    "total_chars": sum(lengths),
}
```
Lines 67-74: Return a dict with:
- count: Number of examples
- mean_length: Average text length in characters
- median_length: Middle value (less sensitive to outliers)
- min_length: Shortest text
- max_length: Longest text
- total_chars: Sum of all text lengths

---

## Design Decisions

1. **Return tuple**: (is_valid, error_message) pattern allows both boolean checking and error reporting.
2. **Error counting**: Tracks error types and frequencies, not just pass/fail. Useful for debugging data quality issues.
3. **Length thresholds**: 50-100,000 chars for CPT, 5+ chars for SFT instructions. Based on empirical observation of Arabic text corpora.
4. **Lazy numpy import**: Avoids adding numpy to the module-level imports (which would slow down any import of this module).
