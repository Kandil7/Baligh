#!/usr/bin/env python3
"""Run SFT training for Baligh-1.5B v0."""

import argparse

import yaml
from datasets import load_from_disk

from baligh.training import train_sft
from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run SFT training for Baligh-1.5B")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to prepared SFT data")
    parser.add_argument("--output-dir", type=str, default="training/sft", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-data", type=str, default=None, help="Evaluation data path")
    parser.add_argument(
        "--base-model", type=str, default=None, help="Path to base model (after CPT)"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    args = parser.parse_args()
    setup_logging()
    logger.info(f"Loading training data from {args.data_dir}")
    train_dataset = load_from_disk(args.data_dir)
    eval_dataset = None
    if args.eval_data:
        logger.info(f"Loading eval data from {args.eval_data}")
        eval_dataset = load_from_disk(args.eval_data)
    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    logger.info("Starting SFT training...")
    result = train_sft(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
        base_model_path=args.base_model,
        config=config,
    )
    logger.info("SFT training completed!")
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
