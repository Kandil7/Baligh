# merge_lora.py - Merge LoRA Script

**Path**: `src/scripts/merge_lora.py` (28 lines)

## Purpose

CLI entry point for merging LoRA adapters into the base model.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --base-model | str | Yes | - | Path to base model |
| --adapter-path | str | Yes | - | Path to LoRA adapter |
| --output-dir | str | Yes | - | Output directory for merged model |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Load base model in full precision (load_in_4bit=False)
4. Load LoRA adapter onto base model
5. Merge adapters into base weights
6. Save merged model

---

## Usage Examples

```bash
# Merge SFT adapters into CPT model
python -m src.scripts.merge_lora \
    --base-model training/cpt/final \
    --adapter-path training/sft/final \
    --output-dir release/baligh-1.5b-v0-instruct
```

---

## Important Notes

- Base model must be loaded in **full precision** (not 4-bit) for merging
- The merged model is a standard AutoModelForCausalLM (no PEFT wrapper)
- Merged model can be quantized to GGUF/AWQ/GPTQ for deployment
