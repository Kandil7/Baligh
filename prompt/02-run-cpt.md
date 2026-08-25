---
description: Run Continued Pretraining (CPT) on Qwen3-1.7B-Base using the prepared CPT data.
agent: ai-engineer
model: ollama/qwen3.5:9b
---

# Goal

Run CPT training on Qwen3-1.7B-Base using the Arrow-format data from `data/train_ready/cpt/`.

# Pre-requisites

- `data/train_ready/cpt/` must exist (run prompt `00-prepare-data` first).

# Steps to Execute

1. Read `configs/cpt/cpt-stage1.yaml` — this is the quick proof-of-concept config (1,000 steps).
2. Read `src/scripts/run_cpt.py` to understand the CLI arguments.
3. Run stage 1 (proof of pipeline):
   ```bash
   uv run python -m src.scripts.run_cpt \
     --data-dir data/train_ready/cpt \
     --config configs/cpt/cpt-stage1.yaml \
     --output-dir training/cpt
   ```
4. Monitor training via TensorBoard:
   ```bash
   uv run tensorboard --logdir training/cpt
   ```
5. After stage 1 completes, verify:
   - `training/cpt/final/` contains adapter weights
   - `training/cpt/checkpoint-*/` exists for auto-resume
   - Loss curve shows downward trend
6. For full training (50K steps), switch to `cpt-stage2.yaml`:
   ```bash
   uv run python -m src.scripts.run_cpt \
     --data-dir data/train_ready/cpt \
     --config configs/cpt/cpt-stage2.yaml \
     --output-dir training/cpt
   ```
7. To resume from the latest checkpoint after interruption:
   ```bash
   uv run python -m src.scripts.run_cpt \
     --data-dir data/train_ready/cpt \
     --config configs/cpt/cpt-stage2.yaml \
     --output-dir training/cpt \
     --resume
   ```

# RTX 5000 16GB Considerations

- The configs use `per_device_train_batch_size=2` + `gradient_accumulation_steps=4` = effective batch 8.
- `packing=true` for CPT (concatenate short texts) — saves VRAM and speeds up training.
- `load_in_4bit=true` with NF4 quantization — ~4GB for 1.7B weights.
- If OOM occurs: reduce `max_seq_length` from 2048 to 1024 in `configs/base/model.yaml`.

# Constraints

- Do NOT change `learning_rate` (2e-4) without documenting the reason.
- Do NOT change the dataset mixing ratios without re-running data preparation.
- Save a copy of the config used alongside the checkpoint.
- Log peak VRAM usage in `docs/training/cpt-log.md`.
