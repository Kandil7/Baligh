# structured.py — Complete Line-by-Line Explanation

**File**: `src/baligh/inference/structured.py` (32 lines)
**Purpose**: Extract structured JSON from model responses with retry logic.

---

## Imports (Lines 1-8)

Line 1: Module docstring.

Line 3: `json` for JSON parsing.

Line 4: `re` for regex (not currently used but imported).

Lines 5-6: Type hints.

Line 7: Import `TextGenerator`.

Line 8: Import logging.

---

## StructuredOutput Class (Lines 11-28)

### Constructor (Lines 12-13)

Lines 12-13: Initialize with TextGenerator.

### extract_json (Lines 15-28)

```python
    def extract_json(self, prompt, schema=None, max_retries=3):
```

Lines 15-16: Extract JSON from model response.

Lines 17-18: Docstring.

Line 19: Append JSON instruction to prompt.

Lines 20-27: Retry loop:
- Generate with low temperature (0.1) for determinism
- Try to parse JSON
- If schema provided, validate (placeholder)
- On parse failure, log warning and retry
- After max_retries, raise ValueError

---

## extract_json Convenience Function (Lines 30-32)

Lines 30-32: Module-level convenience function.
