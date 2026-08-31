#!/usr/bin/env python3
"""
Prepare Data Script for Baligh-1.7B
Downloads, standardizes, cleans, mixes, and formats Hugging Face datasets.

Pipeline:
- CPT: ArabicWeb24 (70%) + ArabicText-Large (20%) + The Arabic Pile (10%) [+ Islamic cycle]
- SFT: CIDAR (40%) + evol-instruct-arabic (35%) + Gazelle (15%) + Summarization (10%)
  — every source is projected onto the canonical {instruction, input, output}
  schema BEFORE mixing, so heterogeneous originals cannot break the merge.

Usage:
    python -m src.scripts.prepare_data --stage cpt --clean --deduplicate
    python -m src.scripts.prepare_data --stage sft --clean
    python -m src.scripts.prepare_data --stage both --clean --no-streaming --deduplicate
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict

from baligh.data import (
    get_cleaning_pipeline,
    get_cpt_formatter,
    get_sft_formatter,
    load_cpt_datasets,
    load_eval_datasets,
    load_sft_datasets,
    mix_cpt_datasets,
    mix_sft_datasets,
    standardize_cpt_dataset,
    standardize_sft_dataset,
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

    VALIDATION_SAMPLE = 2000

    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set seed for reproducibility
        set_seed(config.seed)
        # Initialize components
        self.cleaning_pipeline = get_cleaning_pipeline() if config.clean else None

    def _materialize(self, dataset, name: str):
        """Materialize a bounded sample when dedup requires map-style data."""
        limit = self.config.max_samples
        if limit is None:
            raise ValueError(
                f"Deduplication of streaming dataset '{name}' requires a bound. "
                "Pass --max-samples N (materializes the first N examples), "
                "or run with --no-streaming."
            )
        logger.info(f"Materializing first {limit} examples of '{name}' for dedup")
        return Dataset.from_list(list(dataset.take(limit)))

    def prepare_cpt(self) -> DatasetDict:
        """Prepare Continued Pretraining data."""
        logger.info("=" * 60)
        logger.info("PREPARING CPT DATA")
        logger.info("=" * 60)
        # 1. Load CPT datasets
        logger.info("Loading CPT datasets from Hugging Face...")
        cpt_datasets = load_cpt_datasets(streaming=self.config.streaming)
        if self.config.max_samples:
            for name, ds in cpt_datasets.items():
                cpt_datasets[name] = ds.take(self.config.max_samples)
        # 2. Standardize schemas (all sources -> {"text"})
        cpt_datasets = {
            name: standardize_cpt_dataset(ds, name) for name, ds in cpt_datasets.items()
        }
        # 3. Clean datasets
        if self.config.clean:
            logger.info("Cleaning CPT datasets...")
            cleaned = {}
            for name, ds in cpt_datasets.items():
                logger.info(f"  Cleaning {name}...")
                cleaned[name] = self._clean_dataset(ds, text_column="text")
            cpt_datasets = cleaned
        # 4. Deduplicate
        if self.config.deduplicate:
            logger.info("Deduplicating CPT datasets...")
            from baligh.data import deduplicate_dataset

            for name, ds in cpt_datasets.items():
                logger.info(f"  Deduplicating {name}...")
                cpt_datasets[name] = deduplicate_dataset(ds, text_column="text")
        # 5. Validate a SAMPLE (full-stream validation wastes hours of
        # bandwidth on web-scale corpora just to log decorative counts)
        logger.info("Validating CPT datasets...")
        for name, ds in cpt_datasets.items():
            valid, invalid, errors = validate_dataset(
                ds, dataset_type="cpt", max_samples=self.VALIDATION_SAMPLE
            )
            logger.info(f"  {name}: {valid} valid, {invalid} invalid")
            if errors:
                logger.warning(f"  Errors: {errors}")
        # 6. Mix datasets by ratio (strict: declared ratios must be real)
        logger.info("Mixing CPT datasets by ratio...")
        mixed_cpt = mix_cpt_datasets(cpt_datasets, seed=self.config.seed, strict=True)
        # 7. Format for CPT (tokenize, pack)
        logger.info("Formatting CPT data...")
        formatter = get_cpt_formatter()
        formatted = mixed_cpt.map(
            formatter,
            batched=True,
            num_proc=None if self.config.streaming else self.config.num_proc,
            remove_columns=mixed_cpt.column_names,
            desc="Formatting CPT",
        )
        # 8. Split train/eval
        return self._split_dataset(formatted, eval_ratio=0.01)

    def prepare_sft(self) -> DatasetDict:
        """Prepare Supervised Fine-Tuning data."""
        logger.info("=" * 60)
        logger.info("PREPARING SFT DATA")
        logger.info("=" * 60)
        # 1. Load SFT datasets
        logger.info("Loading SFT datasets from Hugging Face...")
        sft_datasets = load_sft_datasets(streaming=False)
        if self.config.max_samples:
            for name, ds in sft_datasets.items():
                sft_datasets[name] = ds.select(range(min(self.config.max_samples, len(ds))))
        # 2. Standardize to canonical {instruction, input, output} BEFORE
        # mixing — summarization (text/summary) and QA sets otherwise break
        # interleave and get silently dropped by the formatter.
        logger.info("Standardizing SFT schemas...")
        sft_datasets = {
            name: standardize_sft_dataset(ds, name) for name, ds in sft_datasets.items()
        }
        # 3. Clean datasets
        if self.config.clean:
            logger.info("Cleaning SFT datasets...")
            cleaned = {}
            for name, ds in sft_datasets.items():
                logger.info(f"  Cleaning {name}...")
                cleaned[name] = self._clean_sft_dataset(ds)
            sft_datasets = cleaned
        # 4. Validate a sample
        logger.info("Validating SFT datasets...")
        for name, ds in sft_datasets.items():
            valid, invalid, errors = validate_dataset(
                ds, dataset_type="sft", max_samples=self.VALIDATION_SAMPLE
            )
            logger.info(f"  {name}: {valid} valid, {invalid} invalid")
            if errors:
                logger.warning(f"  Errors: {errors}")
        # 5. Mix by ratio (strict)
        logger.info("Mixing SFT datasets by ratio...")
        mixed_sft = mix_sft_datasets(sft_datasets, seed=self.config.seed, strict=True)
        # 6. Format for SFT (chat template + response-only loss masking)
        logger.info("Formatting SFT data...")
        formatter = get_sft_formatter()
        formatted = mixed_sft.map(
            formatter,
            batched=True,
            num_proc=None if self.config.streaming else self.config.num_proc,
            remove_columns=mixed_sft.column_names,
            desc="Formatting SFT",
        )
        # 7. Split train/eval
        return self._split_dataset(formatted, eval_ratio=0.02)

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
            return {text_column: pipeline.clean_batch(texts)}

        from datasets import IterableDataset

        kwargs: dict = {"batched": True}
        if isinstance(dataset, IterableDataset):
            # IterableDataset.map() supports neither num_proc nor desc.
            pass
        else:
            kwargs["num_proc"] = self.config.num_proc
            kwargs["desc"] = f"Cleaning {text_column}"
        # The cleaning pipeline is row-dropping (it removes empty/invalid
        # texts), so any sibling column would go out of sync with `text`.
        cols = dataset.column_names
        if cols:
            kwargs["remove_columns"] = [c for c in cols if c != text_column]
        return dataset.map(clean_batch, **kwargs)

    def _clean_sft_dataset(self, dataset: Dataset) -> Dataset:
        """Apply cleaning to the canonical SFT columns, preserving rows.

        Row-preserving sanitize (not row-dropping clean) is essential here:
        dropping one column's entry but not its siblings corrupts alignment.
        """
        if self.cleaning_pipeline is None:
            return dataset

        pipeline = self.cleaning_pipeline

        def sanitize_batch(batch):
            for col in ("instruction", "input", "output"):
                if col in batch:
                    batch[col] = [pipeline.sanitize(t) for t in batch[col]]
            return batch

        return dataset.map(
            sanitize_batch,
            batched=True,
            num_proc=self.config.num_proc,
            desc="Cleaning SFT",
        )

    def _split_dataset(self, dataset, eval_ratio: float = 0.01) -> DatasetDict:
        """Split dataset into train/eval."""
        from datasets import IterableDataset

        if isinstance(dataset, IterableDataset):
            # Streaming: take a fixed eval slice, skip it in train.
            eval_size = 1000
            return DatasetDict({"train": dataset.skip(eval_size), "eval": dataset.take(eval_size)})
        split = dataset.train_test_split(test_size=eval_ratio, seed=self.config.seed, shuffle=True)
        return DatasetDict({"train": split["train"], "eval": split["test"]})

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
                "base_model": "unsloth/Qwen3-1.7B-Base",
            },
        }
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {subdir}: {meta['splits']}")

    def push_to_hf(self, dataset_dict: DatasetDict, repo_id: str, subdir: str):
        """Push dataset to Hugging Face Hub."""
        logger.info(f"Pushing {subdir} to {repo_id}...")
        dataset_dict.push_to_hub(
            repo_id,
            config_name=subdir,
            token=self.config.hf_token,
            commit_message=f"Add {subdir} dataset for Baligh-1.7B",
        )
        logger.info(f"Pushed to HF: {repo_id}/{subdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training data for Baligh-1.7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare CPT data with cleaning
  python -m src.scripts.prepare_data --stage cpt --clean --output-dir data/train_ready
  # Prepare SFT data with cleaning
  python -m src.scripts.prepare_data --stage sft --clean --output-dir data/train_ready
  # Full non-streaming pipeline with deduplication
  python -m src.scripts.prepare_data --stage both --clean --deduplicate --no-streaming
  # Quick test with 1000 samples per dataset
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream large corpora (default: true; use --no-streaming to materialize)",
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
        default="Kandil7/Baligh-1.7B-data",
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
    setup_logging(log_level=args.log_level)
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
        if args.stage == "eval":
            eval_data = preparator.prepare_eval()
            preparator.save(eval_data, "eval")
        logger.info("=" * 60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 60)
    except Exception as e:
        logger.exception(f"Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
