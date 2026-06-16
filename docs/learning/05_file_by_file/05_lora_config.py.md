# lora_config.py — Complete Line-by-Line Explanation

**File**: `src/baligh/training/lora_config.py` (35 lines)
**Purpose**: Creates PEFT LoRAConfig and BitsAndBytesConfig from project config dataclasses.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: `LoraConfig`, `TaskType` from PEFT.

Line 4: `BitsAndBytesConfig` from transformers.

Line 5: `torch` for dtype selection.

Line 6: Import project config factories.

---

## create_lora_config (Lines 8-21)

```python
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
```

Lines 8-21: Convert project LoRAConfig to PEFT LoraConfig.
- Use custom config or default
- Map all fields to PEFT equivalents
- task_type=CAUSAL_LM for decoder-only models
- target_modules: Convert tuple to list

---

## create_quantization_config (Lines 23-35)

```python
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
```

Lines 23-35: Create BitsAndBytesConfig based on ModelConfig.
- If 4-bit: Create NF4 config with double quant
- If 8-bit: Create simple 8-bit config
- Otherwise: Return None (full precision)
