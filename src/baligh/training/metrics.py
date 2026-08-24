"""Training metrics for Baligh-1.7B v0."""

import torch

from typing import Any

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    """Compute eval_loss + perplexity from an EvalPrediction.

    Note: registering this callback without Trainer's
    ``preprocess_logits_for_metrics`` accumulates full-vocab logits in
    memory (~152k x seq x batch fp32) — pair them if you ever wire it up.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item()
    return {"eval_loss": loss.item(), "perplexity": perplexity}


def compute_perplexity(model, dataloader, device):
    """Token-weighted perplexity over a dataloader.

    Loss is accumulated per LABELLED token (labels != -100), not per
    sequence, so padded/short batches cannot skew the estimate.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    import torch  # local import kept: callers may pass CPU-only envs

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            n_labelled = int((labels != -100).sum().item())
            if n_labelled == 0:
                continue
            # HF models return mean loss over non-ignored tokens of THIS batch.
            total_loss += outputs.loss.item() * n_labelled
            total_tokens += n_labelled

    avg_loss = total_loss / total_tokens if total_tokens > 0 else None
    if avg_loss is None or avg_loss <= 0:
        # Perfect (or empty) loss -> perplexity 1, never infinity.
        return 1.0 if avg_loss is not None and avg_loss == 0 else float("nan")
    return float(torch.exp(torch.tensor(avg_loss)))
