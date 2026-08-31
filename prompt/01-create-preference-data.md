---
description: Create preference data pairs (chosen/rejected) for DPO alignment from the SFT dataset using LLM generation and automated quality filtering.
agent: arabic-data-engineer
model: ollama/qwen3.5:9b-32k
---

# Goal

Generate a preference dataset of **2,000–5,000** (prompt, chosen, rejected) triples for DPO training. The chosen response should be high-quality Arabic; the rejected response should contain identifiable flaws. Below ~1K pairs, sigmoid-DPO tends to overfit the reference margin at beta=0.1.

# Input

Use the SFT data from `data/train_ready/sft/` as the source of prompts/instructions.

# Steps to Execute

1. Read `src/baligh/data/preference.py` to understand the expected format.
2. Read `src/baligh/data/datasets.py` to find SFT dataset sources (CIDAR, evol-instruct-arabic, gazelle, summarization).
3. Write a Python script `src/scripts/create_preference_data.py` that:
   a. Loads the SFT training split from `data/train_ready/sft/`.
   b. Samples 2,000–5,000 instructions (diverse across domains).
   c. For each instruction, generates:
      - **chosen**: a well-formed, accurate Arabic response (use a strong model like Qwen3 or GPT-4o-mini via API).
      - **rejected**: a deliberately degraded response (inject one or more flaws):
        - Factual error or hallucination
        - Missing citation for Quran/Hadith references
        - Mixed languages (code-switch mid-sentence)
        - Truncated or incomplete answer
        - Wrong formatting (e.g., table where list is expected)
   d. Applies automated quality filters:
      - Arabic ratio ≥ 0.3 for both chosen and rejected
      - Chosen length > rejected length (chosen should be more complete)
      - No exact duplicates
      - Rejection must differ from chosen by ≥ 20% token overlap
   e. Validates every row against the project contract before writing:
      ```python
      from baligh.data.preference import validate_preference_example
      ```
      Output columns MUST be `prompt`, `chosen`, `rejected`. The loader also accepts `instruction` as an alias and auto-renames it (`normalize_preference_schema`), but prefer canonical from the start.
   f. Saves to `data/preference/train.jsonl`.
   g. Generates `data/preference/manifest.json` via `get_preference_stats()` plus rejection-type counts.
4. Run the script:
   ```bash
   uv run python -m src.scripts.create_preference_data
   ```
5. Verify the output file exists and has the correct schema.

# Rejection Strategy Details

For Islamic content specifically:
- chosen: accurate Quran/Hadith citation with proper isnad
- rejected: missing attribution or misattributed hadith

For general instruction following:
- chosen: complete, well-structured response
- rejected: partially correct but missing key information

# Constraints

- Do NOT use a weak model for chosen generation — the quality of DPO depends on chosen being genuinely better.
- Do NOT create rejected responses that are offensive — they should be *wrong*, not *harmful*.
- Keep both chosen and rejected in Arabic (or match the instruction language).
- Log how many examples pass/fail each quality filter.
