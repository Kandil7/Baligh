"""Merge LoRA adapters for Baligh-1.5B v0."""

import argparse
from pathlib import Path
from baligh.models import load_base_model, merge_lora
from baligh.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters for Baligh-1.5B")
    parser.add_argument("--base-model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged model")
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("Loading base model from %s" % args.base_model)
    model = load_base_model(model_name=args.base_model, load_in_4bit=False)
    
    logger.info("Merging LoRA from %s" % args.adapter_path)
    merged = merge_lora(model, args.output_dir)
    
    logger.info("Merged model saved to %s" % args.output_dir)

if __name__ == "__main__":
    main()
