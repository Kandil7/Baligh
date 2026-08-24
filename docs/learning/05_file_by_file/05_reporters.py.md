# reporters.py — Complete Line-by-Line Explanation

**File**: `src/baligh/evaluation/reporters.py` (36 lines)
**Purpose**: Generate JSON evaluation reports.

---

## Imports (Lines 1-7)

Line 1: Module docstring.

Line 3: `json` for JSON serialization.

Line 4: `Path` for file paths.

Line 5: `datetime` for timestamps.

Line 6: Import logging.

Line 7: Create logger.

---

## generate_eval_report (Lines 10-25)

```python
def generate_eval_report(results, output_dir, model_name="Baligh-1.7B-v0"):
```

Lines 10-11: Generate a timestamped evaluation report.

Lines 12-13: Docstring.

Line 14: Create output directory if needed.

Lines 16-19: Build report dict with model name, timestamp, and results.

Lines 21-22: Create filename with model name and timestamp.

Lines 23-24: Write JSON to file (ensure_ascii=False for Arabic text).

Line 25: Return the report file path.

---

## save_results (Lines 27-36)

```python
def save_results(results, output_dir, prefix="eval"):
```

Lines 27-28: Generic results saver.

Lines 29-30: Docstring.

Line 31: Create output directory if needed.

Lines 33-34: Create filename with prefix and timestamp.

Lines 35-36: Write JSON to file.

Line 36: Return the results file path.
