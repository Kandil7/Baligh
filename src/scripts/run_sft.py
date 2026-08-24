#!/usr/bin/env python3
"""Run SFT training for Baligh-1.7B."""

import argparse
from pathlib import Path

import yaml
from datasets import DatasetDict, load_from_disk

from baligh.training import train_sft
from baligh.training.checkpoint import CheckpointManager
from baligh.utils.logging import get_logger, setup_logging
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


def split_dataset_dict(obj):
    """load_from_disk returns a DatasetDict for stage directories; pick
    the train/eval splits explicitly instead of passing a dict to Trainer."""
    if isinstance(obj, DatasetDict):
        train = obj["train"]
        eval_ds = obj.get("eval", obj.get("test"))
        return train, eval_ds
    return obj, None


def main():
    parser = argparse.ArgumentParser(description="Run SFT training for Baligh-1.7B")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to prepared SFT data")
    parser.add_argument("--output-dir", type=str, default="training/sft", help="Output directory")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Resume from checkpoint. Pass path or use without value to auto-find latest.",
    )
    parser.add_argument("--eval-data", type=str, default=None, help="Evaluation data path")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="CPT output to continue from (load-bearing: without it SFT "
        "starts from the raw base model)",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    args = parser.parse_args()
    setup_logging()
    set_seed()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    logger.info(f"Loading training data from {args.data_dir}")
    loaded = load_from_disk(str(data_path))
    train_dataset, disk_eval = split_dataset_dict(loaded)

    eval_dataset = disk_eval
    if args.eval_data:
        logger.info(f"Loading eval data from {args.eval_data}")
        eval_loaded = load_from_disk(args.eval_data)
        _, eval_dataset = split_dataset_dict(eval_loaded)

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    resume_from_checkpoint = None
    if args.resume is not None:
        if args.resume == "":
            manager = CheckpointManager(args.output_dir)
            latest = manager.get_latest_checkpoint()
            if latest and manager.validate_checkpoint(latest):
                resume_from_checkpoint = str(latest)
                logger.info(f"Auto-resuming from: {resume_from_checkpoint}")
            else:
                logger.info("No valid checkpoint found, starting from scratch")
        else:
            resume_from_checkpoint = args.resume

    logger.info("Starting SFT training...")
    result = train_sft(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        base_model_path=args.base_model,
        config=config,
    )
    logger.info("SFT training completed!")
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
