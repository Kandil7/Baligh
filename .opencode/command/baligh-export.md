---
description: Merge LoRA adapters, quantize to GGUF/AWQ, generate model card, optionally push to HF.
agent: builder
model: ollama/qwen3.5:9b
---

Load skills: `model-compression`, `huggingface`.

Execute the playbook at `prompt/06-export-model.md` exactly. Requires `training/dpo/final/` from /baligh-dpo.

Never push without an explicit user go-ahead; model card must be generated before any push. Quantized artifacts stay local under D:\AI\Models\baligh\quantized\.

User extra instructions: $ARGUMENTS
