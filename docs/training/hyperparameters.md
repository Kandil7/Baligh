# Hyperparameters Reference

## Model

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-1.7B |
| Parameters | 1.54B |
| Layers | 28 |
| Hidden Size | 2048 |
| Attention Heads | 16 (GQA) |
| Context Length | 32,768 (train: 2048) |
| Vocab Size | 151,936 |

## Quantization (QLoRA)

| Parameter | Value |
|-----------|-------|
| Load in 4-bit | True |
| Quant Type | NF4 |
| Double Quant | True |
| Compute Dtype | bfloat16 |

## LoRA

| Parameter | Value |
|-----------|-------|
| r | 16 |
| alpha | 16 |
| dropout | 0.0 |
| bias | none |
| Target Modules | q,k,v,o,gate,up,down_proj |

## CPT Training

| Parameter | Stage 1 | Stage 2 | Islamic |
|-----------|---------|---------|---------|
| Max Steps | 1,000 | 50,000 | 5,000 |
| Learning Rate | 2e-4 | 2e-4 | 1e-4 |
| Warmup Steps | 100 | 1,000 | 200 |
| Batch Size | 2 | 2 | 2 |
| Grad Accum | 4 | 4 | 4 |
| Eff. Batch | 8 | 8 | 8 |
| Seq Length | 2048 | 2048 | 2048 |
| Packing | True | True | True |

## SFT Training

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Max Steps | 5,000 | 10,000 |
| Epochs | 1.0 | 2.0 |
| Learning Rate | 1e-4 | 5e-5 |
| Warmup Steps | 500 | 500 |
| Batch Size | 2 | 2 |
| Grad Accum | 4 | 4 |
| Seq Length | 2048 | 2048 |
| Packing | False | False |
| Response-only Loss | True | True |

## Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | adamw_torch_fused |
| Beta1 | 0.9 |
| Beta2 | 0.95 |
| Epsilon | 1e-8 |
| Weight Decay | 0.01 |
| Max Grad Norm | 1.0 |

## Mixed Precision

| Parameter | Value |
|-----------|-------|
| Precision | bf16 |
| Gradient Checkpointing | True |
