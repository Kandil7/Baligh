"""Run evaluation for Baligh-1.7B."""

import argparse

from baligh.evaluation import (
    Evaluator,
    generate_eval_report,
    run_cidar_eval,
    run_islamic_qa,
    run_mmlu_arabic,
)
from baligh.utils.logging import get_logger, setup_logging
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for Baligh-1.7B")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--output-dir", type=str, default="eval/results", help="Output directory")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu", "cidar"],
        choices=["mmlu", "cidar", "islamic"],
        help="Benchmarks to run (islamic requires a QA dataset path)",
    )
    parser.add_argument(
        "--islamic-data",
        type=str,
        default=None,
        help="HF dataset id/path with question/answer columns",
    )
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per benchmark")
    args = parser.parse_args()

    setup_logging()
    set_seed()

    logger.info(f"Loading model from {args.model_path}")
    evaluator = Evaluator(args.model_path, args.adapter_path)

    results = {}

    if "mmlu" in args.benchmarks:
        logger.info("Running MMLU-Arabic benchmark...")
        results["mmlu"] = run_mmlu_arabic(evaluator, max_samples=args.max_samples)

    if "cidar" in args.benchmarks:
        logger.info("Running CIDAR evaluation...")
        results["cidar"] = run_cidar_eval(evaluator, max_samples=args.max_samples)

    if "islamic" in args.benchmarks:
        if not args.islamic_data:
            raise SystemExit(
                "--benchmarks islamic requires --islamic-data "
                "(a HF dataset id/path with question/answer columns). "
                "Refusing to silently skip a requested benchmark."
            )
        from datasets import load_dataset

        logger.info(f"Running Islamic QA evaluation on {args.islamic_data}...")
        qa_dataset = load_dataset(args.islamic_data, split="test")
        results["islamic_qa"] = run_islamic_qa(evaluator, qa_dataset, max_samples=args.max_samples)

    logger.info("Generating evaluation report...")
    generate_eval_report(results, args.output_dir)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
