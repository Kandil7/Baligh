"""LoRA and quantization configuration for Baligh-1.7B v0.

Single construction point for PEFT/BNB configs. ``baligh.models.loader``
delegates here; do not build LoraConfig/BitsAndBytesConfig inline elsewhere.
"""

from peft import LoraConfig, TaskType

from baligh.config import get_lora_config, get_model_config
from baligh.utils.hardware import resolve_precision, resolve_torch_dtype
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def create_lora_config(custom_config=None) -> LoraConfig:
    """Build a PEFT LoraConfig from baligh's LoRAConfig defaults."""
    cfg = custom_config or get_lora_config()
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg.target_modules),
        modules_to_save=list(cfg.modules_to_save) if cfg.modules_to_save else None,
        init_lora_weights=cfg.init_lora_weights,
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
    )


def create_quantization_config(compute_dtype=None):
    """Build a BitsAndBytesConfig from ModelConfig defaults.

    Honors the explicit ``bnb_4bit_compute_dtype`` setting, but guards
    against requesting bfloat16 on pre-sm_80 GPUs (falls back to fp16).
    """
    import torch
    from transformers import BitsAndBytesConfig

    mcfg = get_model_config()
    requested = (mcfg.bnb_4bit_compute_dtype or "").lower()
    if compute_dtype is None:
        from baligh.utils.hardware import get_device_capability

        capability = get_device_capability()
        if requested in ("bf16", "bfloat16"):
            if capability is not None and capability < (8, 0):
                logger.info(
                    f"bnb_4bit_compute_dtype=bfloat16 unsupported on sm_{capability[0]}"
                    f"{capability[1]}; falling back to float16"
                )
                compute_dtype = torch.float16
            else:
                compute_dtype = torch.bfloat16
        elif requested in ("fp16", "float16"):
            compute_dtype = torch.float16
        else:
            compute_dtype = resolve_torch_dtype(resolve_precision("auto"))

    if mcfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=mcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=mcfg.bnb_4bit_use_double_quant,
        )
    if mcfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    logger.info("No quantization requested (load_in_4bit/8bit both False)")
    return None
