---
description: Validate the full Baligh training chain end-to-end (CPT+SFT+DPO+eval, 10 steps each) before any real run.
agent: builder
model: ollama/qwen3.5:9b-32k
---

Load skills: `debugging`, `qlora`.

Execute the playbook at `prompt/07-smoke-test.md` exactly. It validates data prep, CPT→SFT→DPO handoffs, and evaluation on tiny samples.

User extra instructions: $ARGUMENTS
