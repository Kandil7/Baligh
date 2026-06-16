# Training Pipeline Documentation

## Overview

Two-stage training: Continued Pretraining (CPT) -> Supervised Fine-Tuning (SFT)

## Stage 1: Continued Pretraining (CPT)

### Objective
Adapt Qwen2.5-1.5B Base to Arabic/Islamic domain

### Configuration
- Base model: unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit
- Method: QLoRA 4-bit (NF4, double quant)
- LoRA: r=16, alpha=16, dropout=0
- Target modules: all attention + MLP
- Sequence length: 2048 (extendable to 32K)
- Packing: True (for efficiency)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 |
| Batch size (per device) | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Warmup steps | 1000 |
| Max steps | 50,000 |
| LR scheduler | Cosine |
| Weight decay | 0.01 |
| Optimizer | adamw_torch_fused |
| Max grad norm | 1.0 |
| Mixed precision | bf16 |
| Gradient checkpointing | True |

### Stages

#### Stage 1: Pipeline Validation (1,000 steps)
- Quick run to verify pipeline works
- Single dataset or small mix
- Config: configs/cpt/cpt-stage1.yaml

#### Stage 2: Full CPT (50,000 steps)
- Full dataset mix
- Config: configs/cpt/cpt-stage2.yaml

#### Islamic Cycle (5,000 steps)
- Separate run on Islamic datasets
- Lower LR: 1e-4
- Config: configs/cpt/cpt-islamic.yaml

## Stage 2: Supervised Fine-Tuning (SFT)

### Objective
Convert base-adapted model to instruction-following assistant

### Configuration
- Base: CPT-trained model
- Method: QLoRA 4-bit (same as CPT)
- Sequence length: 2048
- Packing: False (preserve conversation structure)
- Response-only loss: True

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (stage1), 5e-5 (stage2) |
| Batch size (per device) | 2 |
| Gradient accumulation | 4 |
| Warmup steps | 500 |
| Max steps | 10,000 |
| LR scheduler | Cosine |
| Weight decay | 0.01 |
| Mixed precision | bf16 |

### Stages

#### Stage 1: Initial SFT (5,000 steps)
- Standard mix
- Config: configs/sft/sft-stage1.yaml

#### Stage 2: Optimized SFT (10,000 steps)
- Higher Islamic QA weight
- Lower LR
- Config: configs/sft/sft-stage2.yaml

## Running Training



## Monitoring

- WandB: loss, learning rate, grad norm, memory
- TensorBoard: same metrics
- Logs: training/logs/
- Checkpoints: every 500 steps (CPT) / 250 steps (SFT)
