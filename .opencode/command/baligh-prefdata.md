---
description: Generate 2000-5000 chosen/rejected Arabic preference pairs for DPO from SFT prompts.
agent: arabic-data-engineer
model: ollama/iKhalid/ALLaM:7b
---

Load skills: `data-pipeline`, `arabic-nlp`, `prompt-engineering`.

Execute the playbook at `prompt/01-create-preference-data.md` exactly. Output contract: `prompt/chosen/rejected` JSONL validated via `baligh.data.preference.validate_preference_example`.

Islamic-content rejection strategy must follow its section (isnad/attribution degradation taxonomy). Do not rely on this model alone for religious rulings — flag rows for scholarly review.

If `ollama/iKhalid/ALLaM:7b` is unavailable, ask the user which strong Arabic-capable model to use for chosen-generation before writing weak pairs.

User extra instructions: $ARGUMENTS
