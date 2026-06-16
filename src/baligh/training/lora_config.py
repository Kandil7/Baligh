"""LoRA and quantization configurations for Baligh-1.5B v0."""

from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import torch
from baligh.config import get_lora_config, get_model_config, get_quantization_config

def create_lora_config(custom_config=None):
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

def create_quantization_config(custom_config=None):
    qcfg = custom_config or get_quantization_config()
    mcfg = get_model_config()
    if mcfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if mcfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
            bnb_4bit_quant_type=mcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=mcfg.bnb_4bit_use_double_quant,
        )
    elif mcfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None
