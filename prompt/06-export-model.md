---
description: Merge LoRA adapters, quantize to GGUF/AWQ, and export the final model.
agent: builder
model: ollama/qwen3.5:9b
---

# Goal

Merge the LoRA adapters into the base model, quantize for deployment, and optionally push to Hugging Face.

# Pre-requisites

- `training/dpo/final/` must exist (run prompt `04-run-dpo` first).

# Steps to Execute

1. Read `src/scripts/merge_lora.py` and `src/scripts/quantize.py`.
2. Merge LoRA adapters into the base model:
   ```bash
   uv run python -m src.scripts.merge_lora \
     --base-model unsloth/Qwen3-1.7B-Base \
     --adapter-path training/dpo/final \
     --output-dir training/merged
   ```
3. Verify the merged model exists at `training/merged/`.
4. Quantize to GGUF (for llama.cpp / Ollama deployment):
   ```bash
   uv run python -m src.scripts.quantize \
     --model-path training/merged \
     --format gguf \
     --quantization q4_k_m \
     --output-dir training/quantized
   ```
5. Optionally quantize to AWQ (for vLLM deployment):
   ```bash
   uv run python -m src.scripts.quantize \
     --model-path training/merged \
     --format awq \
     --output-dir training/quantized_awq
   ```
6. Generate a model card:
   ```bash
   uv run python -m src.scripts.generate_model_card \
     --model-name Baligh-1.7B-v0 \
     --output-dir training/merged
   ```
7. Push to Hugging Face (requires HF token):
   ```bash
   uv run python -m src.scripts.push_to_hf \
     --model-path training/merged \
     --repo-id Kandil7/Baligh-1.7B-v0 \
     --token $HF_TOKEN
   ```

# Constraints

- Do NOT push quantized models to HF — only the full-precision merged model.
- Do NOT skip the model card generation — every published model needs documentation.
- Save the merged model to `D:\AI\Models\baligh\merged\` for local use.
- Save quantized models to `D:\AI\Models\baligh\quantized\` for deployment.
