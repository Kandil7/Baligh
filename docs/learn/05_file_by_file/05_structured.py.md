# structured.py - Structured Output

**Path**: `src/baligh/inference/structured.py` (32 lines)

## Purpose

Extract structured JSON from model responses with retry logic.

---

## StructuredOutput (lines 11-28)

### extract_json (lines 15-28)

1. Appends JSON instruction to prompt
2. Generates with low temperature (0.1) for determinism
3. Attempts JSON parsing
4. Retries up to max_retries (default 3) on parse failure
5. Raises ValueError if all retries fail

Note: Schema validation is a placeholder (line 24: `if schema: pass`).

---

## extract_json (lines 30-32)

Module-level convenience function.
