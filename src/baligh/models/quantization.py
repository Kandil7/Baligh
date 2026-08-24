"""Quantization configurations for model export."""

from dataclasses import dataclass

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GGUFConfig:
    """GGUF quantization configuration."""

    quantization: str = "q4_k_m"  # q4_k_m, q5_k_m, q8_0, f16, f32
    output_dir: str = "./gguf"


@dataclass(frozen=True, slots=True)
class AWQConfig:
    """AWQ quantization configuration."""

    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"  # gemm, gemv
    output_dir: str = "./awq"


@dataclass(frozen=True, slots=True)
class GPTQConfig:
    """GPTQ quantization configuration."""

    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    output_dir: str = "./gptq"


def quantize_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    convert_script: str | None = None,
) -> None:
    """Convert a merged HF model to GGUF using llama.cpp's converter.

    Args:
        model_path: Path to a MERGED fp16 model directory.
        output_path: Output GGUF file path.
        quantization: llama.cpp quantization type (q4_k_m, q5_k_m, q8_0...).
        convert_script: Optional path to llama.cpp ``convert_hf_to_gguf.py``.

    Raises:
        RuntimeError: If the conversion tooling is unavailable or fails.
    """
    import shutil
    import subprocess
    from pathlib import Path

    logger.info(f"Quantizing to GGUF: {quantization}")

    # Two-step pipeline: (1) convert HF -> f16 GGUF, (2) requantize to target.
    # `llama-cpp-python` does NOT ship a converter module; the real tooling is
    # llama.cpp's convert_hf_to_gguf.py + llama-quantize binaries.
    script: Path | None = None
    if convert_script:
        script = Path(convert_script)
    else:
        found = shutil.which("convert_hf_to_gguf.py") or shutil.which(
            "convert_hf_to_gguf"
        )
        script = Path(found) if found else None

    if script is None or not script.exists():
        raise RuntimeError(
            "GGUF export requires llama.cpp's convert_hf_to_gguf.py. "
            "Clone llama.cpp and pass --convert-script /path/to/convert_hf_to_gguf.py, "
            "or install it on PATH. See docs/evaluation or release workflow for the "
            "pinned checkout used in CI."
        )

    intermediate = str(Path(output_path).with_suffix(".f16.gguf"))
    cmd = [
        "python",
        str(script),
        str(model_path),
        "--outfile",
        intermediate,
        "--outtype",
        "f16",
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"GGUF conversion failed: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin and quantization != "f16":
        cmd = [quantize_bin, intermediate, output_path, quantization]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"GGUF requantization failed: {result.stderr}")
            raise RuntimeError(f"GGUF requantization failed: {result.stderr}")
        Path(intermediate).unlink(missing_ok=True)
    else:
        if quantization != "f16":
            logger.warning(
                "llama-quantize binary not found; keeping full-precision "
                f"f16 GGUF at {intermediate} (requested {quantization})"
            )

    logger.info(f"GGUF saved to: {output_path}")


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "gemm",
) -> None:
    """Quantize model using AWQ.

    Args:
        model_path: Path to model.
        output_path: Output directory.
        bits: Quantization bits.
        group_size: Group size.
        zero_point: Whether to use zero point.
        version: AWQ version.
    """
    from autoawq import AutoAWQForCausalLM  # type: ignore
    from transformers import AutoTokenizer

    logger.info(f"Quantizing to AWQ: {bits}-bit, group_size={group_size}")

    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": version,
    }

    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"AWQ model saved to: {output_path}")


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    calibration_texts: list[str] | None = None,
) -> None:
    """Quantize a model using GPTQ.

    Args:
        model_path: Path to model directory.
        output_path: Output directory.
        bits: Quantization bits.
        group_size: Group size.
        desc_act: Whether to use desc_act.
        calibration_texts: REQUIRED sample texts for calibration. GPTQ
            without calibration produces an invalid artifact — this
            function refuses to run without data (use domain Arabic text,
            e.g. samples from the CPT mix).
    """
    # Fail BEFORE importing heavy optional deps: no calibration data means
    # this call can never succeed.
    if not calibration_texts:
        raise ValueError(
            "GPTQ quantization requires calibration_texts (>=128 diverse "
            "samples recommended). Load them from the CPT corpus before export."
        )

    from auto_gptq import AutoGPTQForCausalLM  # type: ignore
    from transformers import AutoTokenizer

    logger.info(f"Quantizing to GPTQ: {bits}-bit, group_size={group_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config={
            "bits": bits,
            "group_size": group_size,
            "desc_act": desc_act,
        },
    )

    # Calibration MUST run before saving, or the output is not quantized.
    examples = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
        for t in calibration_texts
    ]
    model.quantize(examples)

    model.save_quantized(output_path, use_safetensors=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"GPTQ model saved to: {output_path}")
