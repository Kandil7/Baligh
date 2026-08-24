<h1 align="center">Baligh-1.7B v0</h1>

<p align="center">
  <img src="docs/assets/baligh-logo.png" alt="Baligh Logo" width="400">
</p>

<p align="center">
  <strong>الفصاحية والذكاء</strong> — Fluency and Intelligence
</p>

<p align="center">
  <a href="https://github.com/Kandil7/Baligh-1.7B/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.11+-yellow.svg" alt="Python">
  </a>
  <a href="https://huggingface.co/Kandil7/Baligh-1.7B">
    <img src="https://img.shields.io/badge/HuggingFace-Model-orange.svg" alt="HuggingFace">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg" alt="PyTorch">
  </a>
</p>

<p align="center">
  Arabic-first LLM with Islamic knowledge specialization, built on Qwen3-1.7B Base.
</p>

---

## Overview

Baligh-1.7B v0 is a lightweight Arabic language model (1.7B total parameters, 1.4B non-embedding) optimized for:

- **Formal Arabic (Fusha)** — fluency and natural quality
- **Islamic knowledge** — Quran, Hadith, Fiqh, Tafsir
- **Instruction following** — Arabic-first chat and instruction tuning
- **Structured output** — JSON extraction and structured responses

## Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | unsloth/Qwen3-1.7B-Base (1.7B params, 28 layers, GQA 16Q/8KV, hidden 2048) |
| **Native Context** | 32,768 tokens (architecture) — v0 trains at a 2,048-token window; no RoPE scaling yet |
| **Training** | Continued Pretraining (CPT) → Supervised Fine-Tuning (SFT) |
| **Method** | QLoRA 4-bit NF4 (r=16, alpha=16); library path = transformers + PEFT, Unsloth in Colab notebooks |
| **Precision** | fp16 on Turing GPUs (T4, RTX 5000), bf16 auto-selected on Ampere+; sdpa attention with flash_attention_2 auto-upgrade on sm_80+ |
| **Tokenizer** | Qwen3 tokenizer (~152K vocab, unchanged for v0) |

## Training Pipeline

### Stage 1: Continued Pretraining (CPT)

| Parameter | Value |
|-----------|-------|
| Datasets | ArabicWeb24 (70%), ArabicText-Large (20%), Arabic Pile (10%) |
| Islamic Cycle | Hadith collections, Quran QA, Quran texts |
| Tokens | ~20-50B Arabic tokens |
| Steps | 50,000 |
| Learning Rate | 2e-4 (cosine scheduler) |
| Config | `configs/cpt/cpt-stage2.yaml` |

### Stage 2: Supervised Fine-Tuning (SFT)

| Parameter | Value |
|-----------|-------|
| Datasets | CIDAR (40%), evol-instruct-arabic (35%), Gazelle (15%), Summarization (10%) |
| Loss | Completion-only (prompt tokens masked with -100) |
| Examples | ~100K-500K instruction examples |
| Steps | 10,000 |
| Learning Rate | 1e-4 (cosine scheduler) |
| Config | `configs/sft/sft-stage2.yaml` |

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- transformers >= 4.51 (Qwen3 architecture support)
- 8GB+ VRAM (recommended 12GB+)

### GPU Requirements

All defaults work on Turing (sm_75): precision falls back to fp16 and
attention to sdpa automatically when bf16 / FlashAttention-2 are unsupported.

| GPU | VRAM | CPT Time | SFT Time | Cost |
|-----|------|----------|----------|------|
| T4 (Free Colab) | 16GB | ~4 hrs | ~2 hrs | Free |
| L4 (Colab Pro) | 24GB | ~2 hrs | ~1 hr | $10/mo |
| A100 40GB | 40GB | ~1 hr | ~30 min | $1-2/hr |

### Installation

```bash
# Clone the repository
git clone https://github.com/Kandil7/Baligh-1.7B.git
cd Baligh-1.7B

# Install base dependencies
make install

# Or install with all dependencies
make install-dev
```

### Data Preparation

```bash
# Prepare CPT data (~20 min)
make data-cpt

# Prepare SFT data (sources are standardized to a canonical schema before mixing)
make data-sft
```

### Training

```bash
# Run CPT training (auto-resumes from latest checkpoint)
make train-cpt

# Run SFT training from the CPT output (--base-model is load-bearing)
make train-sft
```

### Checkpoint Management

Training saves FULL-STATE checkpoints (weights + optimizer + scheduler +
trainer state + RNG) so every resume continues exactly where it stopped:

```bash
# Auto-resume from latest valid checkpoint
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --resume

# Resume from specific checkpoint
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt \
  --resume training/cpt/checkpoint-5000

# List all checkpoints with metadata
python -c "
from baligh.training.checkpoint import CheckpointManager
m = CheckpointManager('training/cpt')
for cp in m.list_checkpoints():
    print(f\"Step {cp['step']:>6} | Loss: {cp.get('loss', 'N/A')}\")
"
```

Each checkpoint directory contains:
- Model weights (LoRA adapter) — `adapter_model.safetensors`
- Optimizer/scheduler state (`optimizer.pt`, `scheduler.pt`)
- `trainer_state.json` + RNG state (exact mid-run continuation)
- `checkpoint_metadata.json` (step, epoch, loss, learning rate, timestamp)
- Automatic cleanup keeps the last 3 manager-saved checkpoints
  (HF-native checkpoints are rotated by Trainer's own `save_total_limit`)
- Graceful Ctrl+C handling (reentrancy-safe save before exit)

### Evaluation

```bash
# Run benchmarks (greedy decoding by default — reproducible numbers)
make eval

# Run specific benchmarks
python -m src.scripts.run_eval --model-path training/sft/final \
  --benchmarks mmlu cidar --max-samples 200
```

Metrics are Arabic-aware: ROUGE uses an Arabic-inclusive tokenizer,
BLEU uses sacrebleu's `tokenize="ar"`, exact match normalizes alef/yaa
variants, diacritics, punctuation and digit forms.

### Inference

```python
from baligh.inference import TextGenerator, ChatBot

# Single generation
generator = TextGenerator("training/sft/final")
response = generator.generate("ما هي عاصمة مصر؟", temperature=0)  # greedy

# Multi-turn chat (history windowed automatically)
chat = ChatBot("training/sft/final")
response = chat.chat("اكتب لي قصيدة قصيرة عن القدس")
```

## Google Colab (Free GPU)

Run the full pipeline on Google Colab with free T4 GPUs (these notebooks use
Unsloth's fused kernels):

| Notebook | Purpose |
|----------|---------|
| `colab_full_pipeline_hf.ipynb` | **Complete pipeline** (recommended) |
| `colab_cpt_hf.ipynb` | CPT training only |
| `colab_sft_hf.ipynb` | SFT training only |
| `colab_eval_hf.ipynb` | Evaluation only |

**Features:**
- Auto-resume from checkpoints on Colab disconnect
- GPU verification on startup
- Checkpoint listing and management
- Push to HuggingFace Hub

## Docker

```bash
# Build and run all services
make docker-up

# Or build individual images (non-root user, .dockerignore enforced)
make docker-cpt    # CPT training
make docker-sft    # SFT training
make docker-eval   # Evaluation
```

## Configuration

| Config | Purpose |
|--------|---------|
| `configs/base/model.yaml` | Model architecture settings (unsloth/Qwen3-1.7B-Base) |
| `configs/base/hardware.yaml` | Hardware requirements |
| `configs/cpt/cpt-stage1.yaml` | CPT quick proof (1K steps) |
| `configs/cpt/cpt-stage2.yaml` | CPT full training (50K steps) |
| `configs/cpt/cpt-islamic.yaml` | Islamic corpus cycle |
| `configs/sft/sft-stage1.yaml` | SFT initial (5K steps) |
| `configs/sft/sft-stage2.yaml` | SFT full training (10K steps) |
| `configs/eval/eval-config.yaml` | Evaluation settings (greedy default) |

## Project Structure

```
Baligh/
├── src/baligh/               # Core library (6 subpackages)
│   ├── config.py             # Pydantic env config + frozen dataclasses
│   ├── data/                 # Registry, clean, standardize, mix, format, validate
│   ├── models/               # Capability-gated loading, tokenizer, quantization
│   ├── training/             # CPT/SFT trainers (plain HF Trainer), checkpoint manager
│   ├── evaluation/           # Arabic-aware benchmarks/metrics, human eval, reporters
│   ├── inference/            # TextGenerator, ChatBot, StructuredOutput
│   └── utils/                # Logging, memory, hardware caps, distributed, seeding
├── src/scripts/              # 8 CLI entry points
├── src/modal/                # Serverless training + authenticated inference endpoints
├── configs/                  # YAML configurations (8 files)
├── colab_cli/                # 4 Unsloth-based Colab notebooks + CLI runner
├── docker/                   # 3 Dockerfiles + docker-compose
├── tests/                    # Unit test suite
├── docs/                     # Documentation
└── requirements/             # Dependency groups
```

## Release Workflow

### Model Artifacts (HF Hub: `Kandil7/Baligh-1.7B`)

| Path | Description |
|------|-------------|
| `cpt/` | CPT checkpoint (LoRA adapter) |
| `sft/` | SFT checkpoint (LoRA adapter) |
| `instruct/` | Merged instruct model (FP16) |
| `gguf/` | GGUF quantized (q4_k_m, q8_0) |
| `eval/` | Evaluation results & reports |

### Release Steps

```bash
# 1. Merge LoRA adapters (loads base in fp16, attaches adapter, merges)
python -m src.scripts.merge_lora \
  --base-model unsloth/Qwen3-1.7B-Base \
  --adapter-path training/sft/final \
  --output-dir release/baligh-1.7b-v0-instruct

# 2. Quantize to GGUF (requires llama.cpp convert_hf_to_gguf.py)
python -m src.scripts.quantize \
  --model-path release/baligh-1.7b-v0-instruct \
  --output-dir release/baligh-1.7b-v0-instruct-gguf \
  --method gguf --quantization q4_k_m \
  --convert-script /path/to/llama.cpp/convert_hf_to_gguf.py

# 3. Generate model card
python -m src.scripts.generate_model_card \
  --output release/baligh-1.7b-v0-instruct/README.md \
  --mmlu-score 0.XX --cidar-rouge 0.XX \
  --islamic-rouge 0.XX --perplexity XX.X

# 4. Push to HF Hub
python -m src.scripts.push_to_hf \
  --model-path release/baligh-1.7b-v0-instruct \
  --repo-id Kandil7/Baligh-1.7B
```

## GitHub Actions (CI/CD)

All workflows declare least-privilege `permissions:`; secrets are scoped to
the steps that need them.

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR | Lint, typecheck, test (80% coverage gate) |
| `cpt-training.yml` | manual | CPT training on self-hosted GPU runner |
| `sft-training.yml` | manual | SFT training on self-hosted GPU runner |
| `release.yml` | manual | Merge, quantize, push to HF Hub, GH release |

## Development

```bash
make lint       # Run linter (ruff)
make format     # Format code (black + ruff)
make typecheck  # Type checking (mypy)
make test       # Run tests with coverage
```

## Dependencies

| File | Purpose |
|------|---------|
| `requirements/base.txt` | Core: torch>=2.6, transformers>=4.51, accelerate, peft, datasets |
| `requirements/training.txt` | Training extras: deepspeed, flash-attn, unsloth (PyPI-pinned, notebooks) |
| `requirements/eval.txt` | Evaluation: rouge-score, bert-score, sacrebleu |
| `requirements/deployment.txt` | Deployment: fastapi, uvicorn |
| `requirements/dev.txt` | Dev: pytest, ruff, mypy, pre-commit |

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{baligh2026,
  title={Baligh-1.7B: Arabic-first LLM with Islamic Knowledge Specialization},
  author={Baligh Team},
  year={2026},
  url={https://github.com/Kandil7/Baligh-1.7B}
}
```

## Links

- **Model Hub**: [Kandil7/Baligh-1.7B](https://huggingface.co/Kandil7/Baligh-1.7B)
- **GitHub**: [Kandil7/Baligh-1.7B](https://github.com/Kandil7/Baligh-1.7B)
- **Documentation**: [docs/learning/](docs/learning/)
- **Issues**: [GitHub Issues](https://github.com/Kandil7/Baligh-1.7B/issues)
