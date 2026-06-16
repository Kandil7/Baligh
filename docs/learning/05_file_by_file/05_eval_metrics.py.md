# eval metrics.py — Complete Line-by-Line Explanation

**File**: `src/baligh/evaluation/metrics.py` (48 lines)
**Purpose**: Standard NLP evaluation metrics for Arabic text generation.

---

## Imports (Lines 1-8)

Line 1: Module docstring.

Line 3: `numpy` for numerical operations.

Line 4: `rouge_scorer` for ROUGE metrics.

Line 5: `corpus_bleu` from sacrebleu for BLEU metrics.

Line 6: `score` from bert_score for semantic similarity.

Line 7: Import logging.

Line 8: Create logger.

---

## compute_perplexity (Lines 11-26)

```python
def compute_perplexity(model, dataloader, device):
```

Lines 11-12: Compute perplexity over a dataloader.

Line 13: Set model to eval mode.

Lines 14-15: Initialize accumulators.

Line 16: Import torch inside function.

Lines 17-24: Iterate over batches:
- Move data to device
- Forward pass with labels
- Accumulate loss * batch_size
- Count tokens

Lines 25-26: Compute average loss and return exp(loss) as perplexity.

---

## compute_rouge (Lines 28-35)

```python
def compute_rouge(predictions, references):
```

Lines 28-29: Compute ROUGE-1, ROUGE-2, ROUGE-L.

Line 30: Create ROUGE scorer with stemmer.

Lines 31-32: Initialize score lists.

Lines 33-34: Score each prediction-reference pair.

Line 35: Return average F-measure for each ROUGE variant.

---

## compute_bleu (Lines 37-40)

```python
def compute_bleu(predictions, references):
```

Lines 37-38: Compute corpus-level BLEU score.

Line 39: Format references as list of lists.

Line 40: Compute and return BLEU score.

---

## compute_bert_score (Lines 42-44)

```python
def compute_bert_score(predictions, references, lang="ar"):
```

Lines 42-43: Compute BERTScore F1 for Arabic.

Line 44: Return precision, recall, F1 averages.

---

## compute_exact_match (Lines 46-48)

```python
def compute_exact_match(predictions, references):
```

Lines 46-47: Compute exact string match ratio.

Line 48: Return matches / total (after stripping whitespace).
