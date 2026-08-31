---
description: Run DPO alignment on the SFT checkpoint using preference data.
agent: ai-engineer
model: ollama/qwen3.5:9b-32k
---

Load skill: `qlora`.

Execute the playbook at `prompt/04-run-dpo.md` exactly. Requires `training/sft/final/` from /baligh-sft and `data/preference/train.jsonl` from /baligh-prefdata.

Always pass `--base-model training/sft/final`. If OOM: reduce --max-length first (768), then consider reference_free=true in config (objective change, not just memory). Log peak VRAM to docs/training/dpo-log.md.

User extra instructions: $ARGUMENTS
