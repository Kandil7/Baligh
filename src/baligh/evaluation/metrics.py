"""Evaluation metrics for Baligh-1.5B v0."""

import numpy as np

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    import torch

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


def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references, strict=False):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    return {k: np.mean(v) for k, v in scores.items()}


def compute_bleu(predictions, references):
    from sacrebleu import corpus_bleu

    refs = [[r] for r in references]
    bleu = corpus_bleu(predictions, refs)
    return bleu.score


def compute_bert_score(predictions, references, lang="ar"):
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}


def compute_exact_match(predictions, references):
    matches = sum(
        1 for p, r in zip(predictions, references, strict=False) if p.strip() == r.strip()
    )
    return matches / len(predictions) if predictions else 0.0
