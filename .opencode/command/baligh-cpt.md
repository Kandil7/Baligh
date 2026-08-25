---
description: Run Continued Pretraining on Qwen3-1.7B-Base (stage1 proof, then stage2 full).
agent: ai-engineer
model: ollama/qwen3.5:9b
---

Load skills: `qlora`, `cuda-pytorch`.

Execute the playbook at `prompt/02-run-cpt.md` exactly. Requires `data/train_ready/cpt/` from /baligh-prepare.

Defaults: start with `configs/cpt/cpt-stage1.yaml` (1000-step proof) before stage2. fp16 only on this Turing GPU — never bf16. Log peak VRAM to docs/training/cpt-log.md.

User extra instructions: $ARGUMENTS
