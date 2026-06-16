# benchmarks.py — Complete Line-by-Line Explanation

**File**: `src/baligh/evaluation/benchmarks.py` (58 lines)
**Purpose**: Run specific benchmarks against an Evaluator instance. Each benchmark tests a different capability.

---

## Imports (Lines 1-7)

Line 1: Module docstring.

Line 3: `load_dataset` from HF datasets.

Line 4: Import `Evaluator` class.

Line 5: Import evaluation metrics (ROUGE, BLEU, exact match).

Line 6: Import logging.

Line 7: Create logger.

---

## run_mmlu_arabic (Lines 10-24)

```python
def run_mmlu_arabic(evaluator, max_samples=100):
```

Lines 10-11: Run MMLU Arabic benchmark. Multiple-choice knowledge test.

Lines 12-13: Log the benchmark.

Lines 14-15: Load MMLU Arabic test split.

Lines 16-23: Iterate over examples:
- Build prompt: question + choices formatted as "A. choice1 B. choice2..."
- Generate response
- Extract first character as prediction
- Check if prediction matches gold answer
- Collect results

Line 24: Compute accuracy and return results.

---

## run_cidar_eval (Lines 26-43)

```python
def run_cidar_eval(evaluator, max_samples=100):
```

Lines 26-27: Run CIDAR evaluation. Arabic instruction-following.

Lines 28-29: Log the benchmark.

Lines 30-31: Load CIDAR test split.

Lines 32-42: Iterate over examples:
- Build prompt from instruction (with optional input)
- Generate response
- Collect predictions and references
- Compute ROUGE and BLEU scores

Line 43: Return ROUGE scores, BLEU score, predictions, and references.

---

## run_islamic_qa (Lines 45-58)

```python
def run_islamic_qa(evaluator, dataset, max_samples=100):
```

Lines 45-46: Run Islamic QA evaluation. Domain knowledge test.

Lines 47-48: Log the benchmark.

Lines 49-57: Iterate over examples:
- Build prompt from question or instruction
- Generate response
- Collect predictions and references
- Compute ROUGE and exact match scores

Line 58: Return ROUGE scores, exact match, predictions, and references.
