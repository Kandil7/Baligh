---
description: Quick smoke test — run a 10-step training loop across CPT, SFT, and DPO to verify the full pipeline works end-to-end.
agent: builder
model: ollama/qwen3.5:9b-32k
---

# Goal

Verify the entire training pipeline works end-to-end with minimal compute before committing to full training runs.

# Steps to Execute

1. Read `scripts/smoke_test_10step.py` to understand the existing smoke test.
2. Prepare minimal data:
   ```bash
   uv run python -m src.scripts.prepare_data --stage both --clean --max-samples 100 --output-dir data/smoke_test
   ```
3. Run the existing smoke test:
   ```bash
   uv run python scripts/smoke_test_10step.py
   ```
4. If the smoke test passes, run a manual 10-step CPT:
   ```bash
   uv run python -m src.scripts.run_cpt \
     --data-dir data/smoke_test/cpt \
     --output-dir training/smoke_cpt
   ```
5. Run a manual 10-step SFT (using the CPT output):
   ```bash
   uv run python -m src.scripts.run_sft \
     --data-dir data/smoke_test/sft \
     --base-model training/smoke_cpt/final \
     --output-dir training/smoke_sft
   ```
6. Create a tiny preference dataset (8 examples) and run 10-step DPO:
   ```bash
   uv run python -c "import json,pathlib; rows=[{'prompt':f'su-al {i}','chosen':'jawab sahih mufassil wa mufid','rejected':'jawab khati qasir'} for i in range(8)]; p=pathlib.Path('data/smoke_test'); p.mkdir(parents=True,exist_ok=True); (p/'preference.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')"
   uv run python -m src.scripts.run_dpo \
     --data-dir data/smoke_test/preference.jsonl \
     --base-model training/smoke_sft/final \
     --output-dir training/smoke_dpo
   ```
7. Run a quick eval:
   ```bash
   uv run python -m src.scripts.run_eval \
     --model-path training/smoke_dpo/final \
     --benchmarks mmlu \
     --max-samples 10 \
     --output-dir docs/evaluation/smoke
   ```
8. Clean up smoke test artifacts:
   ```bash
   Remove-Item -Recurse -Force data/smoke_test, training/smoke_cpt, training/smoke_sft, training/smoke_dpo
   ```

# What This Validates

- Data loading from all sources
- Schema standardization (CPT and SFT)
- Cleaning pipeline execution
- Formatter tokenization (ChatML + loss masking)
- Model loading with QLoRA
- CPT training loop
- SFT training loop (with base_model handoff)
- DPO training loop (with preference data)
- Evaluation pipeline
- Checkpoint save/resume

# Constraints

- Use `--max-samples 100` to keep data small.
- If any step fails, fix the issue before proceeding to full training.
- This is a validation step, not a training step — don't tune hyperparameters here.
