"""Merge LoRA adapters into a deployable fp16 model for Baligh-1.7B.

Correct sequence (each step load-bearing):
1. Load base in fp16/bf16 ” adapters CANNOT merge into bnb-4bit layers.
2. Wrap with PeftModel.from_pretrained(adapter_path) ” without this there
   is nothing to merge and merge_and_unload() crashes.
3. merge_and_unload() -> save_pretrained(safetensors) + tokenizer.
"""

import argparse
from pathlib import Path

from baligh.models import load_base_model, load_lora_model, merge_lora
from baligh.models.tokenizer import get_tokenizer
from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters for Baligh-1.7B")
    parser.add_argument("--base-model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for merged model"
    )
    args = parser.parse_args()

    setup_logging()

    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    if not Path(args.base_model).exists() and "/" not in args.base_model:
        raise FileNotFoundError(f"Base model path not found: {args.base_model}")

    logger.info(f"Loading base model in full precision from {args.base_model}")
    # Full precision is REQUIRED: 4-bit layers cannot absorb merged weights.
    base = load_base_model(model_name=args.base_model, load_in_4bit=False)

    logger.info(f"Attaching adapters from {args.adapter_path}")
    peft_model = load_lora_model(base, args.adapter_path, is_trainable=False)

    output_dir = Path(args.output_dir)
    merged = merge_lora(peft_model, save_path=str(output_dir))

    tokenizer = get_tokenizer(args.base_model)
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"Merged model saved to {output_dir}")
    return merged


if __name__ == "__main__":
    main()
