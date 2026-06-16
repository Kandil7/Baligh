# benchmarks.py - Benchmark Runners

**Path**: `src/baligh/evaluation/benchmarks.py` (58 lines)

## Purpose

Run specific benchmarks against an Evaluator instance.

---

## run_mmlu_arabic (lines 10-24)

MMLU Arabic multiple-choice:
1. Loads FreedomIntelligence/MMLU_Arabic test split
2. Formats each question with choices
3. Extracts first character of response as prediction
4. Computes accuracy (exact match of prediction vs gold answer)

---

## run_cidar_eval (lines 26-43)

CIDAR instruction-following:
1. Loads arbml/CIDAR test split
2. Sends instruction (with optional input) as prompt
3. Computes ROUGE and BLEU scores against reference outputs

---

## run_islamic_qa (lines 45-58)

Islamic QA evaluation:
1. Uses a provided dataset (not loaded from HF)
2. Sends question as prompt
3. Computes ROUGE and exact match scores
