"""Evaluation package for Baligh-1.5B v0."""

from baligh.evaluation.benchmarks import run_cidar_eval, run_islamic_qa, run_mmlu_arabic
from baligh.evaluation.evaluator import Evaluator, evaluate_model
from baligh.evaluation.human_eval import EvaluationRubric, HumanEvaluator
from baligh.evaluation.metrics import (
    compute_bert_score,
    compute_bleu,
    compute_perplexity,
    compute_rouge,
)
from baligh.evaluation.reporters import generate_eval_report, save_results

__all__ = [
    "Evaluator",
    "evaluate_model",
    "compute_perplexity",
    "compute_rouge",
    "compute_bleu",
    "compute_bert_score",
    "run_mmlu_arabic",
    "run_cidar_eval",
    "run_islamic_qa",
    "HumanEvaluator",
    "EvaluationRubric",
    "generate_eval_report",
    "save_results",
]
