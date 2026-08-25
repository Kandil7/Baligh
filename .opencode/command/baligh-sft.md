---
description: Run SFT on the CPT checkpoint (stage1, then stage2). --base-model is load-bearing.
agent: ai-engineer
model: ollama/qwen3.5:9b
---

Load skill: `qlora`.

Execute the playbook at `prompt/03-run-sft.md` exactly. Requires `training/cpt/final/` from /baligh-cpt.

CRITICAL: always pass `--base-model training/cpt/final` — without it SFT trains from the raw base model and silently discards all CPT work. Log peak VRAM to docs/training/sft-log.md.

User extra instructions: $ARGUMENTS
