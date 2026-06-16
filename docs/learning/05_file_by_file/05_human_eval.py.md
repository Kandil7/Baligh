# human_eval.py — Complete Line-by-Line Explanation

**File**: `src/baligh/evaluation/human_eval.py` (53 lines)
**Purpose**: Structured human evaluation system with rubric-based scoring.

---

## Imports (Lines 1-5)

Line 1: Module docstring.

Line 3: `dataclass`, `field` for data structures.

Line 4: `List`, `Dict` for type hints.

Line 5: Import `get_eval_config` for default rubric.

---

## EvaluationRubric (Lines 11-21)

```python
@dataclass
class EvaluationRubric:
    correctness: str = "1-5: Factual accuracy of the response"
    clarity: str = "1-5: Clarity and readability of Arabic"
    arabic_quality: str = "1-5: Formal Arabic quality (fusha)"
    usefulness: str = "1-5: Helpfulness for the user query"
    faithfulness: str = "1-5: Faithfulness to source knowledge"
```

Lines 11-16: Dataclass defining the 5 scoring dimensions:
- correctness: Is the response factually accurate?
- clarity: Is the Arabic clear and readable?
- arabic_quality: Is it proper formal Arabic (fusha)?
- usefulness: Does it help the user?
- faithfulness: Is it faithful to source knowledge?

Lines 18-21: Class method to create rubric from config defaults.

---

## HumanEvaluation (Lines 23-29)

```python
@dataclass
class HumanEvaluation:
    prompt: str
    response: str
    scores: Dict[str, int]
    annotator: str
    notes: str = ""
```

Lines 23-29: Dataclass storing a single evaluation:
- prompt: The question/instruction
- response: The model's response
- scores: Dict mapping dimension names to 1-5 scores
- annotator: Who performed the evaluation
- notes: Optional additional comments

---

## HumanEvaluator (Lines 31-53)

Lines 31-33: Initialize with rubric and empty evaluations list.

Lines 36-38: Add a new evaluation.

Lines 40-44: Compute average scores across all evaluations.

Lines 46-52: Export all evaluations to CSV with columns for each rubric dimension.
