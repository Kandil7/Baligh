# Baligh-1.7B on Modal

Run Baligh-1.7B training and inference on Modal's serverless GPU cloud.

## Setup

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Set secrets
modal secret create huggingface-token HF_TOKEN=hf_xxxxx
modal secret create wandb-token WANDB_API_KEY=xxx  # optional
```

## Training

### Prepare Data

```bash
# Download and prepare CPT data
modal run src/modal/train.py::prepare_data --stage cpt

# Download and prepare SFT data
modal run src/modal/train.py::prepare_data --stage sft
```

### CPT Training

```bash
# Quick test (1K steps)
modal run src/modal/train.py --stage cpt --config configs/cpt/cpt-stage1.yaml

# Full training (50K steps)
modal run src/modal/train.py --stage cpt --config configs/cpt/cpt-stage2.yaml

# Auto-resume from latest checkpoint
modal run src/modal/train.py --stage cpt --resume
```

### SFT Training

```bash
# Initial SFT (5K steps)
modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final

# Full SFT (10K steps)
modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final --config configs/sft/sft-stage2.yaml

# Auto-resume
modal run src/modal/train.py --stage sft --resume
```

### Checkpoint Management

```bash
# List all checkpoints
modal run src/modal/train.py::list_checkpoints --stage cpt
modal run src/modal/train.py::list_checkpoints --stage sft
```

## Inference

### CLI Inference

```bash
# Quick inference
modal run src/modal/infer.py --prompt "ما هي عاصمة مصر؟"

# With custom params
modal run src/modal/infer.py --prompt "اكتب قصة قصيرة" --max-new-tokens 1024 --temperature 0.8
```

### Web Endpoint

```bash
# Deploy inference server
modal deploy src/modal/serve.py

# Test locally
modal serve src/modal/serve.py
```

Then access:
- `https://your-username--baligh-generate.modal.run/?prompt=ما+هي+عاصمة+مصر؟`
- `https://your-username--baligh-chat.modal.run/?message=مرحبا`

## GPU Options

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| T4 | 16GB | ~$0.60 | Quick tests, inference |
| A10G | 24GB | ~$1.10 | CPT/SFT training |
| A100 40GB | 40GB | ~$3.40 | Full training |
| A100 80GB | 80GB | ~$4.90 | Large batch training |
| H100 | 80GB | ~$7.50 | Fastest training |

Change GPU in `app.py` or pass `--gpu a100` to Modal commands.

## Costs Estimate

| Task | GPU | Time | Cost |
|------|-----|------|------|
| Data prep | CPU | ~20 min | ~$0.10 |
| CPT stage1 (1K steps) | A10G | ~30 min | ~$0.55 |
| CPT full (50K steps) | A10G | ~4 hrs | ~$4.40 |
| SFT stage1 (5K steps) | A10G | ~1 hr | ~$1.10 |
| SFT full (10K steps) | A10G | ~2 hrs | ~$2.20 |
| Inference | A10G | per request | ~$0.01 |

## Persistent Storage

Modal volumes persist data across runs:
- `/vol/data/` — Training data
- `/vol/training/cpt/` — CPT checkpoints
- `/vol/training/sft/` — SFT checkpoints
- `/vol/release/` — Merged models
- `/vol/.cache/` — HuggingFace cache

Data persists until you delete the volume:
```bash
modal volume delete baligh-training
```

## Architecture

```
Modal Cloud
├── Persistent Volume (baligh-training)
│   ├── data/train_ready/    # Prepared datasets
│   ├── training/cpt/        # CPT checkpoints + metadata
│   ├── training/sft/        # SFT checkpoints + metadata
│   ├── release/             # Merged + quantized models
│   └── .cache/              # HuggingFace cache
│
├── GPU Functions
│   ├── train()              # CPT/SFT training
│   ├── prepare_data()       # Data preparation
│   ├── infer_cli()          # CLI inference
│   ├── generate()           # Web endpoint (GET)
│   └── chat()               # Web endpoint (GET)
│
└── Secrets
    ├── huggingface-token    # HF Hub access
    └── wandb-token          # W&B logging (optional)
```

## Troubleshooting

### Out of Memory
```bash
# Use larger GPU
modal run src/modal/train.py --stage cpt --gpu a100

# Or reduce batch size in config
```

### Slow Downloads
```bash
# Data is cached in volume — first run is slow, subsequent runs are fast
# Force re-download:
modal volume rm baligh-training /vol/data
```

### Checkpoint Not Found
```bash
# List available checkpoints
modal run src/modal/train.py::list_checkpoints --stage cpt

# Or check volume
modal volume ls baligh-training /vol/training/cpt/
```
