# training metrics.py - Training Metrics

**Path**: `src/baligh/training/metrics.py` (34 lines)

## Purpose

Computes training metrics: cross-entropy loss and perplexity.

---

## compute_metrics (lines 9-18)

Standard causal LM metric function for HF Trainer:
1. Shifts logits and labels by 1 position (predict next token)
2. Computes cross-entropy loss
3. Computes perplexity as exp(loss)
4. Returns dict with eval_loss and perplexity

---

## compute_perplexity (lines 20-34)

Standalone perplexity computation over a dataloader:
1. Sets model to eval mode
2. Iterates batches, accumulates loss * batch_size
3. Returns exp(average_loss)

Used for evaluation outside the training loop.
