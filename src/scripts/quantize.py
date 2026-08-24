"""Quantize a merged Baligh-1.7B model for deployment."""

import argparse
from pathlib import Path

from baligh.models import quantize_awq, quantize_gguf, quantize_gptq
from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quantize model for Baligh-1.7B")
    parser.add_argument("--model-path", type=str, required=True, help="Path to MERGED fp16 model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--method", choices=["gguf", "awq", "gptq"], required=True, help="Quantization method"
    )
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="GGUF quantization type")
    parser.add_argument("--bits", type=int, default=4, help="Bits for AWQ/GPTQ")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for AWQ/GPTQ")
    parser.add_argument(
        "--convert-script",
        type=str,
        default=None,
        help="Path to llama.cpp convert_hf_to_gguf.py (required for GGUF)",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Text file with calibration samples, one per line (required for GPTQ)",
    )
    args = parser.parse_args()

    setup_logging()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    if args.method == "gguf":
        logger.info(f"Quantizing to GGUF: {args.quantization}")
        quantize_gguf(
            args.model_path,
            args.output_dir,
            args.quantization,
            convert_script=args.convert_script,
        )
    elif args.method == "awq":
        logger.info(f"Quantizing to AWQ: {args.bits}-bit")
        quantize_awq(args.model_path, args.output_dir, bits=args.bits, group_size=args.group_size)
    elif args.method == "gptq":
        if not args.calibration_data:
            raise SystemExit("GPTQ requires --calibration-data (text file, one sample per line)")
        with open(args.calibration_data, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Quantizing to GPTQ: {args.bits}-bit with {len(texts)} calibration samples")
        quantize_gptq(
            args.model_path,
            args.output_dir,
            bits=args.bits,
            group_size=args.group_size,
            calibration_texts=texts,
        )

    logger.info("Quantization completed!")


if __name__ == "__main__":
    main()
