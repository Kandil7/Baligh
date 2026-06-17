# run_cpt.py - CPT Training Script

**Path**: `src/scripts/run_cpt.py` (41 lines)

## Purpose

CLI entry point for running Continued Pretraining (CPT) on the Baligh model.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --data-dir | str | Yes | - | Path to prepared CPT data |
| --output-dir | str | No | "training/cpt" | Output directory for checkpoints |
| --resume | str | No | None | Resume from checkpoint path |
| --eval-data | str | No | None | Evaluation data path |
| --config | str | No | None | YAML config file path |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Load training data from disk (HF format)
4. Optionally load evaluation data
5. Optionally load YAML config (overrides defaults)
6. Call `train_cpt()` with all parameters
7. Log results

---

## Usage Examples

```bash
# Basic CPT training
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt

# With custom config
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --config configs/cpt/cpt-stage1.yaml

# Resume from checkpoint
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --resume training/cpt/checkpoint-5000

# With evaluation data
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --eval-data data/train_ready/cpt/eval
```
