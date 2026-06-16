# lora_config.py - LoRA and Quantization Config

**Path**: `src/baligh/training/lora_config.py` (35 lines)

## Purpose

Creates PEFT LoRAConfig and BitsAndBytesConfig from the project's config dataclasses.

---

## create_lora_config (lines 8-21)

Converts `LoRAConfig` dataclass to PEFT `LoraConfig`:
- Maps `target_modules` from tuple to list
- Maps `modules_to_save` if present
- Sets `task_type=TaskType.CAUSAL_LM`
- Passes through r, alpha, dropout, bias, init_lora_weights, use_rslora, use_dora

---

## create_quantization_config (lines 23-35)

Creates `BitsAndBytesConfig` based on ModelConfig:
- If `load_in_4bit=True`: Creates 4-bit config with NF4, double quant, bfloat16 compute
- If `load_in_8bit=True`: Creates 8-bit config
- Otherwise: Returns None (full precision)
