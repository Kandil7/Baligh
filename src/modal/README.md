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

# API key for the web endpoints (required before `modal deploy`)
modal secret create baligh-api-key BALIGH_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
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
# Continue from CPT (--base-model is load-bearing)
modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final

# Full SFT with explicit config
modal run src/modal/train.py --stage sft \
  --base-model /vol/training/cpt/final --config configs/sft/sft-stage2.yaml

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

Endpoints are **authenticated**: every request needs your
`BALIGH_API_KEY` value in the `x-api-key` header. Unauthenticated GPU
inference is cost abuse waiting to happen.

### CLI Inference

```bash
modal run src/modal/serve.py --prompt "ما هي عاصمة مصر؟"
```

### Web Endpoints (POST + x-api-key)

```bash
modal deploy src/modal/serve.py

curl -X POST https://<your-workspace>--baligh-1-7b-generate.modal.run/ \
  -H "x-api-key: $BALIGH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ما هي عاصمة مصر؟", "max_new_tokens": 512}'

curl -X POST https://<your-workspace>--baligh-1-7b-chat.modal.run/ \
  -H "x-api-key: $BALIGH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "مرحبا", "history": []}'
```

## GPU Options

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| T4 | 16GB | ~$0.60 | Quick tests, inference |
| A10G | 24GB | ~$1.10 | CPT/SFT training |
| A100 40GB | 40GB | ~$3.40 | Full training |
| H100 | 80GB | ~$7.50 | Fastest training |

Change GPU in `app.py`. Precision and attention adapt automatically:
A10G/A100 get bf16 + flash_attention_2, T4 gets fp16 + sdpa.

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
│   ├── training/cpt/        # CPT checkpoints (full-state, resumable)
│   ├── training/sft/        # SFT checkpoints
│   ├── release/             # Merged + quantized models
│   └── .cache/              # HuggingFace cache
│
├── GPU Functions
│   ├── train()              # CPT/SFT training (subprocess, list args)
│   ├── prepare_data()       # Data preparation
│   ├── infer_cli()          # CLI inference
│   ├── generate()           # Web endpoint (POST, x-api-key)
│   └── chat_endpoint()      # Web endpoint (POST, x-api-key)
│
└── Secrets
    ├── huggingface-token    # HF Hub access
    ├── wandb-token          # W&B logging (optional)
    └── baligh-api-key       # Web endpoint auth (required for deploy)
```

## Troubleshooting

### Out of Memory
Reduce batch size in the stage config, or move to a larger GPU class.

### Slow Downloads
Data is cached in the volume — first run is slow, subsequent runs are fast.

### Checkpoint Not Found
```bash
modal run src/modal/train.py::list_checkpoints --stage cpt
modal volume ls baligh-training /vol/training/cpt/
```

### 401 from web endpoints
The `baligh-api-key` secret is missing on the server or the `x-api-key`
header does not match. Recreate the secret and redeploy.
