# run_sft.py - SFT Training Script

**Path**: `src/scripts/run_sft.py` (43 lines)

## Purpose

CLI entry point for running Supervised Fine-Tuning (SFT) on the Baligh model.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --data-dir | str | Yes | - | Path to prepared SFT data |
| --output-dir | str | No | "training/sft" | Output directory for checkpoints |
| --resume | str | No | None | Resume from checkpoint path |
| --eval-data | str | No | None | Evaluation data path |
| --base-model | str | No | None | Path to base model (after CPT) |
| --config | str | No | None | YAML config file path |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Load training data from disk (HF format)
4. Optionally load evaluation data
5. Optionally load YAML config (overrides defaults)
6. Call `train_sft()` with all parameters
7. Log results

---

## Usage Examples

```bash
# Basic SFT training
python -m src.scripts.run_sft --data-dir data/train_ready/sft

# With custom base model (after CPT)
python -m src.scripts.run_sft --data-dir data/train_ready/sft --base-model training/cpt/final

# With custom config
python -m src.scripts.run_sft --data-dir data/train_ready/sft --config configs/sft/sft-stage1.yaml

# Resume from checkpoint
python -m src.scripts.run_sft --data-dir data/train_ready/sft --resume training/sft/checkpoint-2500
```
