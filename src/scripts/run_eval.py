"""Run evaluation for Baligh-1.5B v0."""

import argparse
from pathlib import Path
from baligh.evaluation import Evaluator, run_mmlu_arabic, run_cidar_eval, run_islamic_qa, generate_eval_report
from baligh.data import load_eval_datasets
from baligh.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for Baligh-1.5B")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--output-dir", type=str, default="eval/results", help="Output directory")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "cidar", "islamic"], help="Benchmarks to run")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per benchmark")
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("Loading model from %s" % args.model_path)
    evaluator = Evaluator(args.model_path, args.adapter_path)
    
    results = {}
    
    if "mmlu" in args.benchmarks:
        logger.info("Running MMLU-Arabic benchmark...")
        results["mmlu"] = run_mmlu_arabic(evaluator, max_samples=args.max_samples)
    
    if "cidar" in args.benchmarks:
        logger.info("Running CIDAR evaluation...")
        results["cidar"] = run_cidar_eval(evaluator, max_samples=args.max_samples)
    
    if "islamic" in args.benchmarks:
        logger.info("Running Islamic QA evaluation...")
        eval_datasets = load_eval_datasets()
        if "islamic_qa_custom" in eval_datasets:
            results["islamic_qa"] = run_islamic_qa(evaluator, eval_datasets["islamic_qa_custom"], max_samples=args.max_samples)
    
    logger.info("Generating evaluation report...")
    generate_eval_report(results, args.output_dir)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
