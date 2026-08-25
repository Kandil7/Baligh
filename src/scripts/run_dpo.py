#!/usr/bin/env python3
"""Run DPO/ORPO training for Baligh-1.7B.

Data contract: prompt/chosen/rejected (string columns). The 'instruction'
alias is auto-renamed to 'prompt'. Tokenization is handled inside TRL.
"""

import argparse
from pathlib import Path

import yaml

from baligh.data.preference import (
    get_preference_stats,
    load_preference_data,
    normalize_preference_schema,
)
from baligh.training import train_dpo
from baligh.training.checkpoint import CheckpointManager
from baligh.utils.logging import get_logger, setup_logging
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run DPO training for Baligh-1.7B")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to preference data (JSONL or dataset)"
    )
    parser.add_argument("--output-dir", type=str, default="training/dpo", help="Output directory")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Resume from checkpoint. Pass path or use without value to auto-find latest.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Held-out preference data; when omitted a 5%% split is carved from --data-dir",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="SFT output to continue from (load-bearing: without it DPO "
        "starts from the raw base model)",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument(
        "--max-length", type=int, default=None, help="Override config max sequence length"
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=None, help="Override config max prompt length"
    )
    args = parser.parse_args()
    setup_logging()
    set_seed()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Preference data not found: {data_path}")

    logger.info(f"Loading preference data from {args.data_dir}")
    train_dataset = normalize_preference_schema(load_preference_data(data_path, split="train"))
    logger.info(f"Preference stats: {get_preference_stats(train_dataset)}")

    if args.eval_data:
        eval_dataset = normalize_preference_schema(load_preference_data(args.eval_data))
    else:
        split = train_dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval")

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    overrides = {
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
    }
    if any(v is not None for v in overrides.values()):
        config = config or {"dpo": {}}
        config.setdefault("dpo", {}).update({k: v for k, v in overrides.items() if v is not None})

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

    if not args.base_model:
        logger.warning(
            "No --base-model given: DPO starts from the RAW base model, "
            "discarding CPT+SFT. Pass the SFT output unless this is intentional."
        )

    logger.info("Starting DPO training...")
    result = train_dpo(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        base_model_path=args.base_model,
        config=config,
    )
    logger.info("DPO training completed!")
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
