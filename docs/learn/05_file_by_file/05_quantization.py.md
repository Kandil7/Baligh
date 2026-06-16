# quantization.py - Model Quantization

**Path**: `src/baligh/models/quantization.py` (147 lines)

## Purpose

Export merged models to quantized formats (GGUF, AWQ, GPTQ) for efficient inference.

---

## Config Dataclasses

- GGUFConfig: quantization type (q4_k_m default), output_dir
- AWQConfig: bits, group_size, zero_point, version, output_dir
- GPTQConfig: bits, group_size, desc_act, output_dir

---

## quantize_gguf (lines 36-69)

Converts model to GGUF using llama.cpp's convert script. Calls `python -m llama_cpp.convert` as a subprocess with model path, output path, and quantization type.

---

## quantize_awq (lines 72-109)

Uses AutoAWQ library:
1. Loads model with AutoAWQForCausalLM
2. Configures quantization (zero_point, group_size, w_bit, version)
3. Calls model.quantize() with tokenizer
4. Saves quantized model and tokenizer

---

## quantize_gptq (lines 112-147)

Uses AutoGPTQ library:
1. Loads model with quantize_config (bits, group_size, desc_act)
2. Saves with safetensors format
3. Saves tokenizer alongside
