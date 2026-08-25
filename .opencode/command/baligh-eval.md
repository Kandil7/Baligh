---
description: Evaluate the aligned model on MMLU-Arabic/CIDAR/Islamic QA and build the comparison report.
agent: model-evaluator
model: ollama/qwen3.5:9b
---

Load skills: `evaluation`, `arabic-nlp`.

Execute the playbook at `prompt/05-evaluate-model.md` exactly. Requires `training/dpo/final/` from /baligh-dpo.

Report measured numbers only — no fabrication, no extrapolation from partial runs. Arabic quality needs human spot-checks alongside automated metrics (see arabic-nlp skill caveats).

User extra instructions: $ARGUMENTS
