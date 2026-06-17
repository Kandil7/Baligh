<h1 align="center">Baligh-1.5B v0</h1>

<p align="center">
  <img src="docs/assets/baligh-logo.png" alt="Baligh Logo" width="400">
</p>

<p align="center">
  <strong>الفصاحية والذكاء</strong> — Fluency and Intelligence
</p>

<p align="center">
  <a href="https://github.com/Kandil7/Baligh-1.5B/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.11+-yellow.svg" alt="Python">
  </a>
  <a href="https://huggingface.co/Kandil7/Baligh-1.5B">
    <img src="https://img.shields.io/badge/HuggingFace-Model-orange.svg" alt="HuggingFace">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg" alt="PyTorch">
  </a>
</p>

<p align="center">
  Arabic-first LLM with Islamic knowledge specialization, built on Qwen2.5-1.5B Base.
</p>

---

## Overview

Baligh-1.5B v0 is a lightweight Arabic language model (1.54B parameters) optimized for:

- **Formal Arabic (Fusha)** — fluency and natural quality
- **Islamic knowledge** — Quran, Hadith, Fiqh, Tafsir
- **Instruction following** — Arabic-first chat and instruction tuning
- **Structured output** — JSON extraction and structured responses

## Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | Qwen2.5-1.5B (1.54B params, 28 layers, GQA, 32K context) |
| **Training** | Continued Pretraining (CPT) → Supervised Fine-Tuning (SFT) |
| **Method** | QLoRA 4-bit (r=16, alpha=16) via Unsloth |
| **Tokenizer** | Qwen2.5 tokenizer (151K vocab, unchanged for v0) |

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
| Datasets | CIDAR (40%), evol-instruct-arabic (35%), Gazelle (10%), Summarization (10%), Islamic QA (5%) |
| Examples | ~100K-500K instruction examples |
| Steps | 10,000 |
| Learning Rate | 1e-4 (cosine scheduler) |
| Config | `configs/sft/sft-stage2.yaml` |

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- 8GB+ VRAM (recommended 12GB+)

### GPU Requirements

| GPU | VRAM | CPT Time | SFT Time | Cost |
|-----|------|----------|----------|------|
| T4 (Free) | 16GB | ~4 hrs | ~2 hrs | Free |
| L4 (Colab Pro) | 24GB | ~2 hrs | ~1 hr | $10/mo |
| A100 40GB | 40GB | ~1 hr | ~30 min | $1-2/hr |
| A100 80GB | 80GB | ~30 min | ~15 min | $2-4/hr |

### Installation

```bash
# Clone the repository
git clone https://github.com/Kandil7/Baligh-1.5B.git
cd Baligh-1.5B

# Install base dependencies
make install

# Or install with all dependencies
make install-dev
```

### Data Preparation

```bash
# Prepare CPT data (~20 min)
make data-cpt

# Prepare SFT data
make data-sft
```

### Training

```bash
# Run CPT training (auto-resumes from latest checkpoint)
make train-cpt

# Run SFT training (auto-resumes from latest checkpoint)
make train-sft
```

### Checkpoint Management

Training automatically saves checkpoints and supports auto-resume:

```bash
# Auto-resume from latest checkpoint
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --resume

# Resume from specific checkpoint
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --resume training/cpt/checkpoint-5000

# List all checkpoints with metadata
python -c "
from baligh.training.checkpoint import CheckpointManager
m = CheckpointManager('training/cpt')
for cp in m.list_checkpoints():
    print(f\"Step {cp['step']:>6} | Loss: {cp.get('loss', 'N/A')}\")
"
```

Each checkpoint saves:
- Model weights (LoRA adapter)
- `checkpoint_metadata.json` (step, epoch, loss, learning rate, timestamp)
- Automatic cleanup (keeps last 3 checkpoints)
- Graceful Ctrl+C handling (saves before exit)

### Evaluation

```bash
# Run all benchmarks
make eval

# Run specific benchmarks
python -m src.scripts.run_eval --model-path training/sft/final --benchmarks mmlu cidar islamic
```

### Inference

```python
from baligh.inference import TextGenerator, ChatBot

# Single generation
generator = TextGenerator("training/sft/final")
response = generator.generate("ما هي عاصمة مصر؟")

# Multi-turn chat
chat = ChatBot("training/sft/final")
response = chat.chat("اكتب لي قصيدة قصيرة عن القدس")
```

## Google Colab (Free GPU)

Run the full pipeline on Google Colab with free T4 GPUs:

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

# Or build individual images
make docker-cpt    # CPT training
make docker-sft    # SFT training
make docker-eval   # Evaluation
```

## Configuration

| Config | Purpose |
|--------|---------|
| `configs/base/model.yaml` | Model architecture settings |
| `configs/base/hardware.yaml` | Hardware requirements |
| `configs/cpt/cpt-stage1.yaml` | CPT quick proof (1K steps) |
| `configs/cpt/cpt-stage2.yaml` | CPT full training (50K steps) |
| `configs/cpt/cpt-islamic.yaml` | Islamic corpus cycle |
| `configs/sft/sft-stage1.yaml` | SFT initial (5K steps) |
| `configs/sft/sft-stage2.yaml` | SFT full training (10K steps) |
| `configs/eval/eval-config.yaml` | Evaluation settings |

## Evaluation

| Benchmark | Metric | Description |
|-----------|--------|-------------|
| MMLU-Arabic | Accuracy | Multiple-choice knowledge |
| CIDAR | ROUGE-L, BLEU | Cultural relevance, instruction following |
| mr-tydi Arabic | Accuracy | Retrieval QA |
| Islamic QA | ROUGE-L, EM | Domain-specific knowledge |
| Arabic Perplexity | PPL | Language modeling quality |
| Human Evaluation | 1-5 rubric | Correctness, clarity, Arabic quality, usefulness, faithfulness |

## Project Structure

```
Baligh/
├── src/baligh/               # Core library (6 subpackages, 25 modules)
│   ├── config.py             # 7 config classes (Pydantic + frozen dataclasses)
│   ├── constants.py          # Paths, model constants, mix ratios
│   ├── data/                 # Data pipeline (clean, mix, format, validate)
│   ├── models/               # Model loading, tokenizer, quantization
│   ├── training/             # CPT/SFT trainers, callbacks, checkpoint manager
│   ├── evaluation/           # Benchmarks, metrics, human eval, reporters
│   ├── inference/            # TextGenerator, ChatBot, StructuredOutput
│   └── utils/                # Logging, memory, distributed, seeding
├── src/scripts/              # 8 CLI entry points
├── configs/                  # YAML configurations (8 files)
├── colab_cli/                # 4 Colab notebooks
├── docker/                   # 3 Dockerfiles + docker-compose
├── tests/                    # 6 unit test files (81 tests)
├── docs/                     # 113+ documentation files
└── requirements/             # 5 dependency groups
```

## Release Workflow

### Model Artifacts (HF Hub: `Kandil7/Baligh-1.5B`)

| Path | Description |
|------|-------------|
| `cpt/` | CPT checkpoint (LoRA adapter) |
| `sft/` | SFT checkpoint (LoRA adapter) |
| `instruct/` | Merged instruct model (FP16) |
| `gguf/` | GGUF quantized (q4_k_m, q8_0) |
| `eval/` | Evaluation results & reports |

### Release Steps

```bash
# 1. Merge LoRA adapters
python -m src.scripts.merge_lora \
  --base-model training/cpt/final \
  --adapter-path training/sft/final \
  --output-dir release/baligh-1.5b-v0-instruct

# 2. Quantize to GGUF
python -m src.scripts.quantize \
  --model-path release/baligh-1.5b-v0-instruct \
  --output-dir release/baligh-1.5b-v0-instruct-gguf \
  --method gguf --quantization q4_k_m

# 3. Generate model card
python -m src.scripts.generate_model_card \
  --output release/baligh-1.5b-v0-instruct/README.md \
  --mmlu-score 0.XX --cidar-rouge 0.XX \
  --islamic-rouge 0.XX --perplexity XX.X

# 4. Push to HF Hub
python -m src.scripts.push_to_hf \
  --model-path release/baligh-1.5b-v0-instruct \
  --repo-id Kandil7/Baligh-1.5B
```

## GitHub Actions (CI/CD)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR | Lint, typecheck, test (80% coverage gate) |
| `cpt-training.yml` | manual | CPT training on GPU runners |
| `sft-training.yml` | manual | SFT training on GPU runners |
| `release.yml` | manual | Merge, quantize, push to HF Hub |

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
| `requirements/base.txt` | Core: transformers, torch, datasets, accelerate, peft, trl |
| `requirements/training.txt` | Training: unsloth, bitsandbytes, wandb, tensorboard |
| `requirements/eval.txt` | Evaluation: rouge-score, bert-score, sacrebleu |
| `requirements/deployment.txt` | Deployment: fastapi, uvicorn |
| `requirements/dev.txt` | Dev: pytest, ruff, mypy, pre-commit |

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{baligh2026,
  title={Baligh-1.5B: Arabic-first LLM with Islamic Knowledge Specialization},
  author={Baligh Team},
  year={2026},
  url={https://github.com/Kandil7/Baligh-1.5B}
}
```

## Links

- **Model Hub**: [Kandil7/Baligh-1.5B](https://huggingface.co/Kandil7/Baligh-1.5B)
- **GitHub**: [Kandil7/Baligh-1.5B](https://github.com/Kandil7/Baligh-1.5B)
- **Documentation**: [docs/learning/](docs/learning/)
- **Issues**: [GitHub Issues](https://github.com/Kandil7/Baligh-1.5B/issues)
