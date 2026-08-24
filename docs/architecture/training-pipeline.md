# Training Pipeline Design

## Overview

Two-stage training: Continued Pretraining (CPT) then Supervised Fine-Tuning (SFT).

## Base Model

- unsloth/Qwen3-1.7B-Base (Base, not Instruct)
- 1.7B total parameters (1.4B non-embedding), 28 layers, GQA 16Q/8KV, 32K native context
- Library training path loads it via transformers + PEFT (Unsloth is used only in Colab notebooks)

## Quantization

- 4-bit NF4 with double quantization
- Compute dtype resolved per GPU: float16 on Turing (sm_75), bfloat16 on Ampere+
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
- Max steps: 10,000
- Learning rate: 5e-5 (lower)
- Load best model at end
- Metric: eval_loss

## Training Infrastructure

### Callbacks (registered on both trainers)
- LoggingCallback: structured log of training metrics
- MemoryCallback: GPU memory stats + cache clear every 100 steps and at epoch end
- Checkpointing is owned by `save_steps` + CheckpointManager (full-state saves:
  weights + optimizer + scheduler + trainer state + RNG), not a callback

### Precision
- Auto-resolved per GPU capability (baligh.utils.hardware):
  fp16 + sdpa on Turing, bf16 + flash_attention_2 on sm_80+

### Monitoring
- Weights & Biases when an API key is configured (training never blocks
  on wandb login — falls back to tensorboard automatically)
- TensorBoard always available
- Log: loss, learning rate, grad norm, memory

## Post-Training

1. Merge LoRA: load base in fp16 -> PeftModel.from_pretrained(adapter) -> merge_and_unload()
   (adapters cannot merge into bnb-4bit layers; the merge script enforces this order)
2. Save full model (safetensors)
3. Quantize: GGUF via llama.cpp convert_hf_to_gguf.py; GPTQ requires calibration data
4. Push to Hugging Face Hub
