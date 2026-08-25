---
description: Run DPO alignment on the SFT checkpoint using the prepared preference data.
agent: ai-engineer
model: ollama/qwen3.5:9b
---

# Goal

Run DPO (Direct Preference Optimization) on the SFT output using the preference data from `data/preference/`.

# Pre-requisites

- `data/preference/train.jsonl` must exist (run prompt `01-create-preference-data` first).
- `training/sft/final/` must exist (run prompt `03-run-sft` first).
- `trl>=0.9` is a core dependency — verify with `uv sync` if imports fail.

# Steps to Execute

1. Read `configs/dpo/dpo-stage1.yaml` — DPO config (1,000 steps).
2. Read `src/scripts/run_dpo.py` to understand the CLI arguments.
3. **Critical**: The `--base-model` flag points at the SFT output. Without it, DPO starts from the raw base model.
4. Run DPO:
   ```bash
   uv run python -m src.scripts.run_dpo \
     --data-dir data/preference/train.jsonl \
     --base-model training/sft/final \
     --config configs/dpo/dpo-stage1.yaml \
     --output-dir training/dpo
   ```
6. Monitor training via TensorBoard:
   ```bash
   uv run tensorboard --logdir training/dpo
   ```
7. Optional: use held-out preference data instead of the automatic 5% split:
   ```bash
   uv run python -m src.scripts.run_dpo \
     --data-dir data/preference/train.jsonl \
     --eval-data data/preference/eval.jsonl \
     --base-model training/sft/final \
     --config configs/dpo/dpo-stage1.yaml \
     --output-dir training/dpo
   ```
8. After training completes, verify:
   - `training/dpo/final/` contains adapter weights
   - Loss curve shows downward trend
   - eval_loss < train_loss
7. To resume after interruption:
   ```bash
   uv run python -m src.scripts.run_dpo \
     --data-dir data/preference/train.jsonl \
     --base-model training/sft/final \
     --config configs/dpo/dpo-stage1.yaml \
     --output-dir training/dpo \
     --resume
   ```

# DPO vs ORPO

The current implementation uses **DPO** (sigmoid loss with a reference term). Key parameters:
- `beta=0.1` — temperature for the DPO loss. Higher = more conservative updates.
- `loss_type=sigmoid` — standard DPO loss. Alternatives: `hinge`, `ipo`.
- `label_smoothing=0.0` — set to 0.1 if training is unstable.

**VRAM note:** with a LoRA/PEFT policy and `ref_model=None`, TRL computes reference logits by disabling adapters — no second copy of the weights is loaded. Setting `reference_free=true` goes further and drops the reference term entirely (ORPO-style objective); use it only if you also want to change the training objective, not just to save memory.

# RTX 5000 16GB Considerations

- Each step runs 4 forward passes through the policy (chosen/rejected × policy/reference paths).
- `per_device_train_batch_size=1` + `gradient_accumulation_steps=8` = effective batch 8.
- `optim=paged_adamw_8bit` — reduces optimizer state memory.
- `max_length=1024`, `max_prompt_length=512` — shorter than SFT to fit in VRAM.
- If OOM occurs: reduce `max_length` first (`--max-length 768`), then consider `reference_free=true`.

# Constraints

- Do NOT change `beta` (0.1) without documenting the reason.
- Do NOT change `--base-model` to point at the raw base model.
- Log peak VRAM usage in `docs/training/dpo-log.md`.
