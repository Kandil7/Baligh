"""Quantization configurations for model export."""

from dataclasses import dataclass
from baligh.config import get_quantization_config
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
) -> None:
    """Quantize model to GGUF format using llama.cpp.
    
    Args:
        model_path: Path to model.
        output_path: Output GGUF file path.
        quantization: Quantization type.
    """
    import subprocess
    import os
    
    logger.info(f"Quantizing to GGUF: {quantization}")
    
    # Convert to GGUF using llama.cpp
    # This requires llama.cpp to be installed
    cmd = [
        "python", "-m", "llama_cpp.convert",
        "--model", model_path,
        "--outfile", output_path,
        "--outtype", quantization,
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"GGUF quantization failed: {result.stderr}")
        raise RuntimeError(f"GGUF quantization failed: {result.stderr}")
    
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
    from autoawq import AutoAWQForCausalLM
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
) -> None:
    """Quantize model using GPTQ.
    
    Args:
        model_path: Path to model.
        output_path: Output directory.
        bits: Quantization bits.
        group_size: Group size.
        desc_act: Whether to use desc_act.
    """
    from auto_gptq import AutoGPTQForCausalLM
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
    
    model.save_quantized(output_path, use_safetensors=True)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"GPTQ model saved to: {output_path}")
