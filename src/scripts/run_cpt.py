#!/usr/bin/env python3
"""Run CPT training for Baligh-1.5B v0."""

import argparse

import yaml
from datasets import load_from_disk

from baligh.training import train_cpt
from baligh.training.checkpoint import CheckpointManager
from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run CPT training for Baligh-1.5B")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to prepared CPT data")
    parser.add_argument("--output-dir", type=str, default="training/cpt", help="Output directory")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Resume from checkpoint. Pass path or use without value to auto-find latest.",
    )
    parser.add_argument("--eval-data", type=str, default=None, help="Evaluation data path")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    args = parser.parse_args()
    setup_logging()

    logger.info(f"Loading training data from {args.data_dir}")
    train_dataset = load_from_disk(args.data_dir)

    eval_dataset = None
    if args.eval_data:
        logger.info(f"Loading eval data from {args.eval_data}")
        eval_dataset = load_from_disk(args.eval_data)

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    resume_from_checkpoint = None
    if args.resume is not None:
        if args.resume == "":
            manager = CheckpointManager(args.output_dir)
            latest = manager.get_latest_checkpoint()
            if latest:
                resume_from_checkpoint = str(latest)
                logger.info(f"Auto-resuming from: {resume_from_checkpoint}")
            else:
                logger.info("No checkpoint found, starting from scratch")
        else:
            resume_from_checkpoint = args.resume

    logger.info("Starting CPT training...")
    result = train_cpt(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        config=config,
    )
    logger.info("CPT training completed!")
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
