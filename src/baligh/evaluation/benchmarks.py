"""Benchmark runners for Baligh-1.5B v0."""

from datasets import load_dataset

from baligh.evaluation.metrics import compute_bleu, compute_exact_match, compute_rouge
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def run_mmlu_arabic(evaluator, max_samples=100):
    """Run MMLU Arabic benchmark."""
    logger.info("Running MMLU-Arabic benchmark")
    dataset = load_dataset("FreedomIntelligence/MMLU_Arabic", split="test")
    results = []
    for example in dataset.select(range(min(max_samples, len(dataset)))):
        choices_text = "\n".join([f"{k}. {v}" for k, v in example["choices"].items()])
        prompt = example["question"] + "\n" + choices_text
        response = evaluator.generate(prompt)
        pred = response.strip()[0] if response.strip() else ""
        correct = pred == example["answer"]
        results.append({"correct": correct, "pred": pred, "gold": example["answer"]})
    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    logger.info("MMLU-Arabic accuracy: %.4f" % accuracy)
    return {"accuracy": accuracy, "results": results}


def run_cidar_eval(evaluator, max_samples=100):
    """Run CIDAR evaluation benchmark."""
    logger.info("Running CIDAR evaluation")
    dataset = load_dataset("arbml/CIDAR", split="test")
    predictions = []
    references = []
    for example in dataset.select(range(min(max_samples, len(dataset)))):
        prompt = example["instruction"]
        if example.get("input"):
            prompt += "\n" + example["input"]
        response = evaluator.generate(prompt)
        predictions.append(response)
        references.append(example["output"])
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    logger.info("CIDAR ROUGE: %s" % rouge)
    logger.info("CIDAR BLEU: %.2f" % bleu)
    return {"rouge": rouge, "bleu": bleu, "predictions": predictions, "references": references}


def run_islamic_qa(evaluator, dataset, max_samples=100):
    """Run Islamic QA evaluation."""
    logger.info("Running Islamic QA evaluation")
    predictions = []
    references = []
    for example in dataset.select(range(min(max_samples, len(dataset)))):
        prompt = example.get("question", example.get("instruction", ""))
        response = evaluator.generate(prompt)
        predictions.append(response)
        references.append(example.get("answer", example.get("output", "")))
    rouge = compute_rouge(predictions, references)
    exact_match = compute_exact_match(predictions, references)
    logger.info("Islamic QA ROUGE: %s" % rouge)
    logger.info("Islamic QA Exact Match: %.4f" % exact_match)
    return {
        "rouge": rouge,
        "exact_match": exact_match,
        "predictions": predictions,
        "references": references,
    }
