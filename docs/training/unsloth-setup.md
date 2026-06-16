# Unsloth Setup Guide

## Overview

Unsloth provides 2x faster fine-tuning with 70% less memory.

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 12-16 GB |
| CUDA | 11.8 | 12.1+ |
| Python | 3.10 | 3.11-3.12 |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 50 GB+ |

## Installation

### Option 1: Colab (Free)
1. Open: https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ
2. Runtime -> Change runtime type -> GPU (T4/L4)
3. Run all cells

### Option 2: Local (Linux/WSL)


### Option 3: uv (Faster)


## Verification



## Common Issues

| Issue | Solution |
|-------|----------|
| OOM | batch_size=1, grad_accum=8 |
| No GPU | conda install cuda-toolkit |
| Triton error | pip install triton or --torch-backend=auto |
| CUDA mismatch | Install PyTorch matching CUDA version |
