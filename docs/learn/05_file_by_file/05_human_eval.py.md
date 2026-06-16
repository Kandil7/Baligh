# human_eval.py - Human Evaluation

**Path**: `src/baligh/evaluation/human_eval.py` (53 lines)

## Purpose

Structured human evaluation system with rubric-based scoring.

---

## EvaluationRubric (lines 11-21)

Dataclass with 5 scoring dimensions:
- correctness: 1-5, factual accuracy
- clarity: 1-5, Arabic readability
- arabic_quality: 1-5, formal Arabic quality (fusha)
- usefulness: 1-5, helpfulness for the query
- faithfulness: 1-5, faithfulness to source knowledge

---

## HumanEvaluation (lines 23-29)

Dataclass storing a single evaluation: prompt, response, scores dict, annotator name, notes.

---

## HumanEvaluator (lines 31-53)

Collects evaluations and computes statistics:
- `add_evaluation()`: Records a new evaluation
- `get_average_scores()`: Computes mean score per dimension
- `export_csv()`: Exports all evaluations to CSV with columns for each rubric dimension
