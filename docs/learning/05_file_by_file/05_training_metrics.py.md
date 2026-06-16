# training metrics.py — Complete Line-by-Line Explanation

**File**: `src/baligh/training/metrics.py` (34 lines)
**Purpose**: Compute training metrics: cross-entropy loss and perplexity.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: `torch` for loss computation.

Line 4: `numpy` for numerical operations.

Line 5: Import logging.

Line 6: Create logger.

---

## compute_metrics (Lines 9-18)

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item()
    return {"eval_loss": loss.item(), "perplexity": perplexity}
```

Lines 9-18: Standard causal LM metric function.
- eval_pred: Tuple of (logits, labels) from the model
- Handle tuple logits (some models return past_key_values)
- Shift logits and labels by 1 (predict next token)
- Compute cross-entropy loss
- Compute perplexity as exp(loss)
- Return dict with eval_loss and perplexity

---

## compute_perplexity (Lines 20-34)

```python
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return np.exp(avg_loss) if avg_loss > 0 else float("inf")
```

Lines 20-34: Standalone perplexity computation.
- Set model to eval mode
- Iterate batches without gradients
- Accumulate loss * batch_size
- Compute average loss
- Return exp(average_loss) as perplexity
