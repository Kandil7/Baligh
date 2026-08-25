---
description: Prepare all CPT and SFT training data via prepare_data.py — download, clean, dedup, mix, format, split.
agent: builder
model: ollama/qwen3.5:9b
---

Load skills: `data-pipeline`, `arabic-nlp`.

Execute the playbook at `prompt/00-prepare-data.md` exactly. It runs the existing `src/scripts/prepare_data.py` pipeline — do NOT write new data scripts.

Default to the bounded smoke variant (`--max-samples 1000`) if the user hasn't confirmed a full download; full corpora exceed 100 GB.

User extra instructions: $ARGUMENTS
