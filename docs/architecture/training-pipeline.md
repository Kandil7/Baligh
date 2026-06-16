# Training Pipeline Design

## Overview

Two-stage training: Continued Pretraining (CPT) then Supervised Fine-Tuning (SFT).

## Base Model

- Qwen2.5-1.5B Base (not Instruct)
- 1.54B parameters, 28 layers, GQA, 32K context
- Unsloth optimized 4-bit: unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit

## Quantization

- 4-bit NF4 with double quantization
- bfloat16 compute dtype
- BitsAndBytesConfig

## LoRA Configuration

- r=16, alpha=16, dropout=0
- Target modules: all attention + MLP (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- bias=none, task_type=CAUSAL_LM
- prepare_model_for_kbit_training with gradient checkpointing

## Stage 1: Continued Pretraining (CPT)

### Stage 1a: Pipeline Validation (cpt-stage1.yaml)
- Max steps: 1,000
- Learning rate: 2e-4
- Warmup: 100 steps
- Goal: Verify pipeline works

### Stage 1b: Full CPT (cpt-stage2.yaml)
- Max steps: 50,000
- Learning rate: 2e-4
- Warmup: 1,000 steps
- Cosine LR schedule
- Batch size: 2, grad accum: 4 (effective: 8)
- Sequence length: 2048
- Packing: enabled
- Optimizer: adamw_torch_fused
- Logging: every 10 steps
- Checkpointing: every 500 steps, keep 3
- Evaluation: every 500 steps

### Islamic CPT Cycle (cpt-islamic.yaml)
- Max steps: 5,000
- Learning rate: 1e-4 (lower)
- Warmup: 200 steps
- Datasets: hadith, quran_qa, quran_md

## Stage 2: Supervised Fine-Tuning (SFT)

### Stage 2a: Initial SFT (sft-stage1.yaml)
- Max steps: 5,000 / 1 epoch
- Learning rate: 1e-4
- Warmup: 500 steps
- Response-only loss: true
- Packing: false

### Stage 2b: Optimized SFT (sft-stage2.yaml)
- Max steps: 10,000 / 2 epochs
- Learning rate: 5e-5 (lower)
- Islamic QA weight: 15%
- Load best model at end
- Metric: eval_loss

## Training Infrastructure

### Callbacks
- LoggingCallback: log metrics to wandb/tensorboard
- MemoryCallback: log GPU memory every 100 steps
- CheckpointCallback: manage checkpoints

### Distributed Training
- DDP with NCCL backend
- Gradient checkpointing enabled
- Mixed precision: bf16

### Monitoring
- Weights & Biases (primary)
- TensorBoard (backup)
- Log: loss, learning rate, grad norm, memory

## Post-Training

1. Merge LoRA: model.merge_and_unload()
2. Save full model (safetensors)
3. Quantize: GGUF (q4_k_m), AWQ (4-bit), GPTQ (4-bit)
4. Push to Hugging Face Hub
