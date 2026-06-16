"""Data preparation script for Baligh-1.5B v0."""

import argparse
from pathlib import Path
from baligh.data import load_cpt_datasets, load_sft_datasets, get_cleaning_pipeline, mix_cpt_datasets, mix_sft_datasets
from baligh.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Baligh-1.5B")
    parser.add_argument("--stage", choices=["cpt", "sft", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default="data/train_ready")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()
    
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.stage in ["cpt", "both"]:
        logger.info("Loading CPT datasets...")
        cpt_datasets = load_cpt_datasets()
        
        if args.clean:
            logger.info("Cleaning CPT datasets...")
            pipeline = get_cleaning_pipeline()
            for name, ds in cpt_datasets.items():
                cpt_datasets[name] = ds.map(pipeline, batched=True, remove_columns=ds.column_names)
        
        logger.info("Mixing CPT datasets...")
        mixed_cpt = mix_cpt_datasets(cpt_datasets)
        
        cpt_output = output_dir / "cpt"
        cpt_output.mkdir(exist_ok=True)
        mixed_cpt.save_to_disk(str(cpt_output))
        logger.info("CPT data saved to %s" % cpt_output)
    
    if args.stage in ["sft", "both"]:
        logger.info("Loading SFT datasets...")
        sft_datasets = load_sft_datasets()
        
        if args.clean:
            logger.info("Cleaning SFT datasets...")
            pipeline = get_cleaning_pipeline()
            for name, ds in sft_datasets.items():
                sft_datasets[name] = ds.map(pipeline, batched=True, remove_columns=ds.column_names)
        
        logger.info("Mixing SFT datasets...")
        mixed_sft = mix_sft_datasets(sft_datasets)
        
        sft_output = output_dir / "sft"
        sft_output.mkdir(exist_ok=True)
        mixed_sft.save_to_disk(str(sft_output))
        logger.info("SFT data saved to %s" % sft_output)
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main()
