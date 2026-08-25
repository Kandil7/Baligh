---
description: Cache raw registry datasets to D:\AI\Datasets\baligh\ as JSONL so reruns never re-download web-scale corpora.
agent: arabic-data-engineer
model: ollama/qwen3.5:9b
---

Load skills: `data-pipeline`, `arabic-nlp`.

Execute the playbook at `prompt/00a-cache-raw-data.md` exactly. Critical constraints it enforces: tashkeel preserved verbatim, registry-driven column projection (quran_qa context/question/answer, mc4_ar config name), manifests with row/char counts.

Start with the smoke run (`--max-samples 50`) unless the user says otherwise.

User extra instructions: $ARGUMENTS
