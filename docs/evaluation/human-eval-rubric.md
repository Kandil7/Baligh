# Human Evaluation Rubric

## Overview

5-point Likert scale for each criterion.

## Rubric

### 1. Correctness (Factual Accuracy)
| Score | Description |
|-------|-------------|
| 1 | Completely incorrect, hallucinated facts |
| 2 | Mostly incorrect, major factual errors |
| 3 | Partially correct, some accurate info |
| 4 | Mostly correct, minor inaccuracies |
| 5 | Fully correct, all facts accurate |

### 2. Clarity (Readability)
| Score | Description |
|-------|-------------|
| 1 | Incomprehensible, garbled Arabic |
| 2 | Hard to understand, broken grammar |
| 3 | Understandable but awkward phrasing |
| 4 | Clear, natural Arabic flow |
| 5 | Exceptionally clear, eloquent |

### 3. Arabic Quality (Fusha)
| Score | Description |
|-------|-------------|
| 1 | Dialect/slang, not fusha |
| 2 | Mix of dialect and fusha |
| 3 | Mostly fusha, some non-standard |
| 4 | Good fusha, proper grammar |
| 5 | Excellent fusha, sophisticated |

### 4. Usefulness
| Score | Description |
|-------|-------------|
| 1 | Not helpful, irrelevant |
| 2 | Slightly helpful, misses point |
| 3 | Moderately helpful, partial answer |
| 4 | Helpful, addresses query well |
| 5 | Extremely helpful, comprehensive |

### 5. Faithfulness
| Score | Description |
|-------|-------------|
| 1 | Fabricated, unfaithful to sources |
| 2 | Mostly unfaithful, major deviations |
| 3 | Partially faithful, some deviations |
| 4 | Mostly faithful, minor deviations |
| 5 | Fully faithful, grounded in knowledge |

## Aggregation

- Average across all criteria per sample
- Average across all samples
- Inter-annotator agreement: Krippendorff alpha
- Target: alpha > 0.7

## Sample Selection

- 100-500 samples
- Stratified across: general QA, Islamic QA, summarization, formatting, refusal
- Blind: annotators don't know model identity
