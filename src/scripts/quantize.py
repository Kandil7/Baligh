"""Quantize model for Baligh-1.5B v0."""

import argparse
from pathlib import Path
from baligh.models import quantize_gguf, quantize_awq, quantize_gptq
from baligh.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Quantize model for Baligh-1.5B")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--method", choices=["gguf", "awq", "gptq"], required=True, help="Quantization method")
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="GGUF quantization type")
    parser.add_argument("--bits", type=int, default=4, help="Bits for AWQ/GPTQ")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for AWQ/GPTQ")
    args = parser.parse_args()
    
    setup_logging()
    
    if args.method == "gguf":
        logger.info("Quantizing to GGUF: %s" % args.quantization)
        quantize_gguf(args.model_path, args.output_dir, args.quantization)
    elif args.method == "awq":
        logger.info("Quantizing to AWQ: %d-bit" % args.bits)
        quantize_awq(args.model_path, args.output_dir, bits=args.bits, group_size=args.group_size)
    elif args.method == "gptq":
        logger.info("Quantizing to GPTQ: %d-bit" % args.bits)
        quantize_gptq(args.model_path, args.output_dir, bits=args.bits, group_size=args.group_size)
    
    logger.info("Quantization completed!")

if __name__ == "__main__":
    main()
