"""Evaluation package for Baligh-1.7B v0."""

from baligh.evaluation.benchmarks import (
    run_cidar_eval,
    run_islamic_qa,
    run_mmlu_arabic,
    run_sunni_benchmark,
)
from baligh.evaluation.evaluator import Evaluator, evaluate_model
from baligh.evaluation.human_eval import EvaluationRubric, HumanEvaluator
from baligh.evaluation.metrics import (
    compute_bert_score,
    compute_bleu,
    compute_exact_match,
    compute_perplexity,
    compute_rouge,
    normalize_for_match,
    tokenize_for_scoring,
)
from baligh.evaluation.reporters import generate_eval_report, save_results

__all__ = [
    "Evaluator",
    "evaluate_model",
    "compute_perplexity",
    "compute_rouge",
    "compute_bleu",
    "compute_bert_score",
    "compute_exact_match",
    "normalize_for_match",
    "tokenize_for_scoring",
    "run_mmlu_arabic",
    "run_cidar_eval",
    "run_islamic_qa",
    "run_sunni_benchmark",
    "HumanEvaluator",
    "EvaluationRubric",
    "generate_eval_report",
    "save_results",
]
