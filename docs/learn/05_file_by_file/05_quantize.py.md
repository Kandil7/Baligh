# quantize.py - Quantization Script

**Path**: `src/scripts/quantize.py` (35 lines)

## Purpose

CLI entry point for quantizing models to GGUF, AWQ, or GPTQ formats.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --model-path | str | Yes | - | Path to model |
| --output-dir | str | Yes | - | Output directory |
| --method | choices | Yes | - | Quantization method (gguf, awq, gptq) |
| --quantization | str | No | "q4_k_m" | GGUF quantization type |
| --bits | int | No | 4 | Bits for AWQ/GPTQ |
| --group-size | int | No | 128 | Group size for AWQ/GPTQ |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Call appropriate quantization function based on --method
4. Log completion

---

## Usage Examples

```bash
# GGUF quantization (for llama.cpp/Ollama)
python -m src.scripts.quantize \
    --model-path release/baligh-1.5b-v0-instruct \
    --output-dir release/baligh-1.5b-v0-instruct-gguf \
    --method gguf \
    --quantization q4_k_m

# AWQ quantization (for vLLM)
python -m src.scripts.quantize \
    --model-path release/baligh-1.5b-v0-instruct \
    --output-dir release/baligh-1.5b-v0-instruct-awq \
    --method awq \
    --bits 4

# GPTQ quantization (for TGI)
python -m src.scripts.quantize \
    --model-path release/baligh-1.5b-v0-instruct \
    --output-dir release/baligh-1.5b-v0-instruct-gptq \
    --method gptq \
    --bits 4
```
