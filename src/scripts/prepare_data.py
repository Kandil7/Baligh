#!/usr/bin/env python3
"""
Prepare Data Script for Baligh-1.5B v0
Downloads, cleans, mixes, and formats Hugging Face datasets for CPT and SFT training.
Implements the exact pipeline from the plan:
- CPT: ArabicWeb24 (70%) + ArabicText-Large (20%) + The Arabic Pile (10%) + Islamic cycle
- SFT: CIDAR (40%) + evol-instruct-arabic (35%) + Gazelle (10%) + Summarization (10%) + Islamic QA (5%)
Usage:
    python -m src.scripts.prepare_data --stage cpt --clean --deduplicate
    python -m src.scripts.prepare_data --stage sft --clean
    python -m src.scripts.prepare_data --stage both --clean --deduplicate
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from baligh.data import (
    get_cleaning_pipeline,
    get_cpt_formatter,
    get_sft_formatter,
    load_cpt_datasets,
    load_eval_datasets,
    load_sft_datasets,
    mix_cpt_datasets,
    mix_sft_datasets,
    validate_dataset,
)
from baligh.utils.logging import get_logger, setup_logging
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


@dataclass
class DataPrepConfig:
    """Configuration for data preparation."""

    stage: str  # "cpt", "sft", "both", "eval"
    output_dir: Path
    clean: bool = False
    deduplicate: bool = False
    streaming: bool = True
    num_proc: int = 8
    seed: int = 42
    max_samples: int | None = None  # For testing
    push_to_hf: bool = False
    hf_repo: str | None = None
    hf_token: str | None = None


class DataPreparator:
    """Main data preparation orchestrator."""

    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set seed for reproducibility
        set_seed(config.seed)
        # Initialize components
        self.cleaning_pipeline = get_cleaning_pipeline() if config.clean else None

    def prepare_cpt(self) -> DatasetDict:
        """Prepare Continued Pretraining data."""
        logger.info("=" * 60)
        logger.info("PREPARING CPT DATA")
        logger.info("=" * 60)
        # 1. Load CPT datasets
        logger.info("Loading CPT datasets from Hugging Face...")
        cpt_datasets = load_cpt_datasets(streaming=self.config.streaming)
        if self.config.max_samples:
            # Limit for testing
            for name, ds in cpt_datasets.items():
                cpt_datasets[name] = ds.take(self.config.max_samples)
        # 2. Clean datasets
        if self.config.clean:
            logger.info("Cleaning CPT datasets...")
            cleaned = {}
            for name, ds in cpt_datasets.items():
                logger.info(f"  Cleaning {name}...")
                cleaned[name] = self._clean_dataset(ds, text_column="text")
            cpt_datasets = cleaned
        # 3. Deduplicate
        if self.config.deduplicate:
            logger.info("Deduplicating CPT datasets...")
            from baligh.data import deduplicate_dataset

            for name, ds in cpt_datasets.items():
                logger.info(f"  Deduplicating {name}...")
                cpt_datasets[name] = deduplicate_dataset(ds, text_column="text")
        # 4. Validate
        logger.info("Validating CPT datasets...")
        for name, ds in cpt_datasets.items():
            valid, invalid, errors = validate_dataset(ds, dataset_type="cpt")
            logger.info(f"  {name}: {valid} valid, {invalid} invalid")
            if errors:
                logger.warning(f"  Errors: {errors}")
        # 5. Mix datasets by ratio
        logger.info("Mixing CPT datasets by ratio...")
        mixed_cpt = mix_cpt_datasets(cpt_datasets, seed=self.config.seed)
        # 6. Format for CPT (tokenize, pack)
        logger.info("Formatting CPT data...")
        formatter = get_cpt_formatter()
        formatted = mixed_cpt.map(
            formatter,
            batched=True,
            num_proc=self.config.num_proc if not self.config.streaming else 1,
            remove_columns=mixed_cpt.column_names,
            desc="Formatting CPT",
        )
        # 7. Split train/eval
        dataset_dict = self._split_dataset(formatted, eval_ratio=0.01)
        return dataset_dict

    def prepare_sft(self) -> DatasetDict:
        """Prepare Supervised Fine-Tuning data."""
        logger.info("=" * 60)
        logger.info("PREPARING SFT DATA")
        logger.info("=" * 60)
        # 1. Load SFT datasets
        logger.info("Loading SFT datasets from Hugging Face...")
        sft_datasets = load_sft_datasets()
        if self.config.max_samples:
            for name, ds in sft_datasets.items():
                sft_datasets[name] = ds.select(range(min(self.config.max_samples, len(ds))))
        # 2. Clean datasets
        if self.config.clean:
            logger.info("Cleaning SFT datasets...")
            cleaned = {}
            for name, ds in sft_datasets.items():
                logger.info(f"  Cleaning {name}...")
                cleaned[name] = self._clean_sft_dataset(ds)
            sft_datasets = cleaned
        # 3. Validate
        logger.info("Validating SFT datasets...")
        for name, ds in sft_datasets.items():
            valid, invalid, errors = validate_dataset(ds, dataset_type="sft")
            logger.info(f"  {name}: {valid} valid, {invalid} invalid")
            if errors:
                logger.warning(f"  Errors: {errors}")
        # 4. Mix datasets by ratio
        logger.info("Mixing SFT datasets by ratio...")
        mixed_sft = mix_sft_datasets(sft_datasets, seed=self.config.seed)
        # 5. Format for SFT (apply chat template)
        logger.info("Formatting SFT data...")
        formatter = get_sft_formatter()
        formatted = mixed_sft.map(
            formatter,
            batched=True,
            num_proc=self.config.num_proc,
            remove_columns=mixed_sft.column_names,
            desc="Formatting SFT",
        )
        # 6. Split train/eval
        dataset_dict = self._split_dataset(formatted, eval_ratio=0.02)
        return dataset_dict

    def prepare_eval(self) -> DatasetDict:
        """Prepare evaluation datasets."""
        logger.info("=" * 60)
        logger.info("PREPARING EVAL DATA")
        logger.info("=" * 60)
        eval_datasets = load_eval_datasets()
        # No cleaning/mixing for eval - keep as-is for consistent benchmarking
        for name, ds in eval_datasets.items():
            logger.info(f"  {name}: {len(ds)} examples")
        return DatasetDict(eval_datasets)

    def _clean_dataset(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """Apply cleaning pipeline to dataset."""
        if self.cleaning_pipeline is None:
            return dataset

        pipeline = self.cleaning_pipeline

        def clean_batch(batch):
            texts = batch[text_column]
            cleaned = pipeline.clean_batch(texts)
            return {text_column: cleaned}

        return dataset.map(
            clean_batch,
            batched=True,
            num_proc=self.config.num_proc if not self.config.streaming else 1,
            desc=f"Cleaning {text_column}",
        )

    def _clean_sft_dataset(self, dataset: Dataset) -> Dataset:
        """Apply cleaning to SFT dataset (instruction, input, output)."""
        if self.cleaning_pipeline is None:
            return dataset

        pipeline = self.cleaning_pipeline

        def clean_batch(batch):
            for col in ["instruction", "input", "output"]:
                if col in batch:
                    batch[col] = pipeline.clean_batch(batch[col])
            return batch

        return dataset.map(
            clean_batch,
            batched=True,
            num_proc=self.config.num_proc,
            desc="Cleaning SFT",
        )

    def _split_dataset(self, dataset: Dataset, eval_ratio: float = 0.01) -> DatasetDict:
        """Split dataset into train/eval."""
        if self.config.streaming:
            # For streaming, take fixed number for eval
            eval_size = 1000
            train_dataset = dataset.skip(eval_size)
            eval_dataset = dataset.take(eval_size)
        else:
            # For regular datasets, use train_test_split
            split = dataset.train_test_split(
                test_size=eval_ratio, seed=self.config.seed, shuffle=True
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        return DatasetDict({"train": train_dataset, "eval": eval_dataset})

    def save(self, dataset_dict: DatasetDict, subdir: str):
        """Save dataset to disk."""
        save_path = self.output_dir / subdir
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {subdir} to {save_path}...")
        dataset_dict.save_to_disk(str(save_path))
        # Save metadata
        meta = {
            "subdir": subdir,
            "splits": {k: len(v) for k, v in dataset_dict.items()},
            "config": {
                "stage": self.config.stage,
                "clean": self.config.clean,
                "deduplicate": self.config.deduplicate,
                "seed": self.config.seed,
            },
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved {subdir}: {meta['splits']}")

    def push_to_hf(self, dataset_dict: DatasetDict, repo_id: str, subdir: str):
        """Push dataset to Hugging Face Hub."""
        logger.info(f"Pushing {subdir} to {repo_id}...")
        dataset_dict.push_to_hub(
            repo_id,
            config_name=subdir,
            token=self.config.hf_token,
            commit_message=f"Add {subdir} dataset for Baligh-1.5B v0",
        )
        logger.info(f"Pushed to HF: {repo_id}/{subdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training data for Baligh-1.5B v0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare CPT data with cleaning
  python -m src.scripts.prepare_data --stage cpt --clean --output-dir data/train_ready
  # Prepare SFT data with cleaning
  python -m src.scripts.prepare_data --stage sft --clean --output-dir data/train_ready
  # Prepare both with deduplication (full pipeline)
  python -m src.scripts.prepare_data --stage both --clean --deduplicate --output-dir data/train_ready
  # Quick test with 1000 samples
  python -m src.scripts.prepare_data --stage cpt --clean --max-samples 1000
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["cpt", "sft", "both", "eval"],
        default="both",
        help="Which data to prepare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/train_ready"),
        help="Output directory for prepared data",
    )
    parser.add_argument("--clean", action="store_true", help="Apply cleaning pipeline")
    parser.add_argument(
        "--deduplicate", action="store_true", help="Apply near-duplicate removal (MinHash LSH)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode for large datasets",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming (load full datasets)",
    )
    parser.add_argument(
        "--num-proc", type=int, default=8, help="Number of processes for parallel processing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples per dataset (for testing)"
    )
    parser.add_argument(
        "--push-to-hf", action="store_true", help="Push prepared data to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="Kandil7/Baligh-1.5B-v0-data",
        help="HF repo ID for pushing data",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None, help="HF token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Setup logging
    setup_logging(log_level=args.log_level)
    # Get HF token from env if not provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    config = DataPrepConfig(
        stage=args.stage,
        output_dir=args.output_dir,
        clean=args.clean,
        deduplicate=args.deduplicate,
        streaming=args.streaming,
        num_proc=args.num_proc,
        seed=args.seed,
        max_samples=args.max_samples,
        push_to_hf=args.push_to_hf,
        hf_repo=args.hf_repo,
        hf_token=hf_token,
    )
    preparator = DataPreparator(config)
    try:
        if args.stage in ["cpt", "both"]:
            cpt_data = preparator.prepare_cpt()
            preparator.save(cpt_data, "cpt")
            if args.push_to_hf:
                preparator.push_to_hf(cpt_data, args.hf_repo, "cpt")
        if args.stage in ["sft", "both"]:
            sft_data = preparator.prepare_sft()
            preparator.save(sft_data, "sft")
            if args.push_to_hf:
                preparator.push_to_hf(sft_data, args.hf_repo, "sft")
        if args.stage in ["eval", "both"]:
            eval_data = preparator.prepare_eval()
            preparator.save(eval_data, "eval")
            if args.push_to_hf:
                preparator.push_to_hf(eval_data, args.hf_repo, "eval")
        logger.info("=" * 60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 60)
    except Exception as e:
        logger.exception(f"Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
