---
description: Run Supervised Fine-Tuning (SFT) on the CPT checkpoint using the prepared SFT data.
agent: ai-engineer
model: ollama/qwen2.5-coder:7b
---

# Goal

Run SFT on the CPT output using the Arrow-format data from `data/train_ready/sft/`.

# Pre-requisites

- `data/train_ready/sft/` must exist (run prompt `00-prepare-data` first).
- `training/cpt/final/` must exist (run prompt `02-run-cpt` first).

# Steps to Execute

1. Read `configs/sft/sft-stage1.yaml` — initial SFT config (5,000 steps).
2. Read `src/scripts/run_sft.py` to understand the CLI arguments.
3. **Critical**: The `--base-model` flag points at the CPT output. Without it, SFT starts from the raw base model and wastes all CPT work.
4. Run stage 1:
   ```bash
   uv run python -m src.scripts.run_sft \
     --data-dir data/train_ready/sft \
     --base-model training/cpt/final \
     --config configs/sft/sft-stage1.yaml \
     --output-dir training/sft
   ```
5. Monitor training via TensorBoard:
   ```bash
   uv run tensorboard --logdir training/sft
   ```
6. After stage 1 completes, verify:
   - `training/sft/final/` contains adapter weights
   - Loss curve shows downward trend
   - eval_loss < train_loss (no overfitting)
7. For optimized training (10K steps), switch to `sft-stage2.yaml`:
   ```bash
   uv run python -m src.scripts.run_sft \
     --data-dir data/train_ready/sft \
     --base-model training/cpt/final \
     --config configs/sft/sft-stage2.yaml \
     --output-dir training/sft
   ```
8. To resume after interruption:
   ```bash
   uv run python -m src.scripts.run_sft \
     --data-dir data/train_ready/sft \
     --base-model training/cpt/final \
     --config configs/sft/sft-stage2.yaml \
     --output-dir training/sft \
     --resume
   ```

# Key Differences from CPT

- `packing=false` for SFT — preserves conversation structure (each instruction is a separate example).
- `response_only_loss=true` — loss computed only on assistant responses, not on the prompt.
- `lr=1e-4` (lower than CPT's 2e-4) — SFT is a finer adjustment.
- `load_best_model_at_end=true` — selects the checkpoint with lowest eval_loss.

# Constraints

- Do NOT run SFT without `--base-model` pointing at CPT output.
- Do NOT change `learning_rate` (1e-4) without documenting the reason.
- Log peak VRAM usage in `docs/training/sft-log.md`.
