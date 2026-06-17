# Baligh-1.5B v0

Arabic-first LLM with Islamic knowledge specialization, built on Qwen2.5-1.5B Base.

## Overview

Baligh-1.5B v0 is a lightweight Arabic language model (1.54B parameters) optimized for:
- **Formal Arabic (Fusha)** fluency and quality
- **Islamic knowledge** (Quran, Hadith, Fiqh, Tafsir)
- **Instruction following** in Arabic
- **Structured output** (JSON extraction)

## Architecture

- **Base Model**: Qwen2.5-1.5B (1.54B params, 28 layers, GQA, 32K context)
- **Training**: Continued Pretraining (CPT) + Supervised Fine-Tuning (SFT)
- **Method**: QLoRA 4-bit (r=16, alpha=16)
- **Tokenizer**: Qwen2.5 tokenizer (unchanged for v0)

## Training Pipeline

### Stage 1: Continued Pretraining (CPT)
- **Datasets**: ArabicWeb24 (70%), ArabicText-Large (20%), The Arabic Pile (10%)
- **Islamic Cycle**: Hadith collections, Quran QA, Quran texts
- **Tokens**: ~20-50B Arabic tokens
- **Config**: `configs/cpt/cpt-stage1.yaml`

### Stage 2: Supervised Fine-Tuning (SFT)
- **Datasets**: CIDAR (40%), evol-instruct-arabic (35%), Gazelle (10%), Summarization (10%), Islamic QA (5%)
- **Examples**: ~100K-500K instruction examples
- **Config**: `configs/sft/sft-stage1.yaml`

## Quick Start

### Prerequisites
- Python 3.11+
- CUDA 12.1+ (for GPU training)
- 8GB+ VRAM (recommended 12GB+)

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
# Prepare CPT data
make data-cpt

# Prepare SFT data
make data-sft
```

### Training

```bash
# Run CPT training
make train-cpt

# Run SFT training
make train-sft
```

### Evaluation

```bash
# Run evaluation
make eval
```

### Inference

```python
from baligh.inference import TextGenerator, ChatBot

# Single generation
generator = TextGenerator("training/sft/final")
response = generator.generate("ما هي عاصمة مصر؟")

# Chat interface
chat = ChatBot("training/sft/final")
response = chat.chat("اكتب لي قصيدة قصيرة عن القدس")
```

## Google Colab (Free GPU)

Run the full pipeline on Google Colab with free T4 GPUs:

```bash
# 1. Open any notebook in colab_cli/ folder in Google Colab
# 2. Run cells sequentially

# Available notebooks:
# - colab_cpt_hf.ipynb      # CPT training only
# - colab_sft_hf.ipynb      # SFT training only
# - colab_eval_hf.ipynb     # Evaluation only
# - colab_full_pipeline_hf.ipynb  # Complete pipeline (recommended)
```

Or use the CLI runner:
```bash
python -m colab_cli.run_colab --notebook full --repo Kandil7/Baligh-1.5B
```

## Docker

### Build Images

```bash
# Build CPT training image
make docker-cpt

# Build SFT training image
make docker-sft

# Build evaluation image
make docker-eval
```

### Run with Docker Compose

```bash
# Start all services
make docker-up

# Or manually
docker-compose -f docker/docker-compose.yml up -d
```

## Configuration

### Base Config
- `configs/base/model.yaml` - Model architecture settings
- `configs/base/hardware.yaml` - Hardware requirements

### Training Configs
- `configs/cpt/cpt-stage1.yaml` - CPT stage 1 (quick proof)
- `configs/cpt/cpt-stage2.yaml` - CPT stage 2 (full training)
- `configs/cpt/cpt-islamic.yaml` - Islamic cycle training
- `configs/sft/sft-stage1.yaml` - SFT stage 1
- `configs/sft/sft-stage2.yaml` - SFT stage 2

### Eval Config
- `configs/eval/eval-config.yaml` - Evaluation settings

## Evaluation

Benchmarks:
- **MMLU-Arabic** - Multiple-choice knowledge
- **CIDAR** - Cultural relevance, instruction following
- **mr-tydi Arabic** - Retrieval QA
- **Islamic QA** - Domain-specific knowledge
- **Arabic Perplexity** - Language modeling quality
- **Human evaluation** - 1-5 rubric (correctness, clarity, Arabic quality, usefulness, faithfulness)

## Release Workflow

### Model Artifacts (HF Hub: `Kandil7/Baligh-1.5B`)

| Path | Description |
|------|-------------|
| `cpt/` | CPT checkpoint (LoRA adapter) |
| `sft/` | SFT checkpoint (LoRA adapter) |
| `instruct/` | Merged instruct model (FP16) |
| `gguf/` | GGUF quantized (q4_k_m, q8_0) |
| `awq/` | AWQ 4-bit quantized |
| `gptq/` | GPTQ 4-bit quantized |
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

# 3. Quantize to AWQ
python -m src.scripts.quantize \
  --model-path release/baligh-1.5b-v0-instruct \
  --output-dir release/baligh-1.5b-v0-instruct-awq \
  --method awq

# 4. Generate model card
python -m src.scripts.generate_model_card \
  --output README.md \
  --mmlu-score 0.XX \
  --cidar-rouge 0.XX \
  --islamic-rouge 0.XX \
  --perplexity XX.X

# 5. Push to HF Hub
python -m src.scripts.push_to_hf \
  --model-dir release/baligh-1.5b-v0-instruct \
  --repo-id Kandil7/Baligh-1.5B \
  --path-in-repo instruct
```

## GitHub Actions (CI/CD)

Automated workflows in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR/push | Lint, typecheck, test |
| `cpt-training.yml` | Manual | CPT training on GPU runners |
| `sft-training.yml` | Manual | SFT training on GPU runners |
| `release.yml` | Tag push | Merge, quantize, push to HF |

### Run Training via GitHub Actions

1. Go to **Actions** → **CPT Training** → **Run workflow**
2. Select config (stage1/stage2/islamic)
3. Wait for completion → artifacts uploaded to HF Hub
4. Repeat for **SFT Training**
5. Create release tag `v0` → triggers **Release** workflow

## Project Structure

```
Baligh/
├── src/baligh/           # Core library
│   ├── config.py         # Configuration management
│   ├── constants.py      # Project constants
│   ├── data/             # Data pipeline
│   ├── training/         # Training loops
│   ├── models/           # Model loading/management
│   ├── evaluation/       # Evaluation framework
│   ├── inference/        # Inference interfaces
│   └── utils/            # Utilities
├── src/scripts/          # CLI entry points
├── configs/              # YAML configurations
├── docker/               # Containerization
├── colab_cli/            # Colab notebooks + runner
├── tests/                # Test suite
├── docs/                 # Documentation
│   └── learning/         # Architecture guides (17 files)
└── requirements/         # Dependencies
```

## Development

### Code Quality

```bash
# Run linter
make lint

# Format code
make format

# Type checking
make typecheck

# Run tests
make test
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

## Dependencies

| File | Purpose |
|------|---------|
| `requirements/base.txt` | Core: transformers, torch, datasets, accelerate, peft, trl |
| `requirements/training.txt` | Training: unsloth, bitsandbytes, wandb, tensorboard |
| `requirements/eval.txt` | Evaluation: lm-eval, rouge-score, bert-score |
| `requirements/quant.txt` | Quantization: llama-cpp-python, autoawq, auto-gptq |
| `requirements/dev.txt` | Dev: pytest, ruff, mypy, pre-commit |

## License

Apache 2.0

## Citation

```bibtex
@software{baligh2024,
  title={Baligh-1.5B: Arabic-first LLM with Islamic Knowledge Specialization},
  author={Baligh Team},
  year={2024},
  url={https://github.com/Kandil7/Baligh-1.5B}
}
```

## Links

- **Model Hub**: https://huggingface.co/Kandil7/Baligh-1.5B
- **GitHub**: https://github.com/Kandil7/Baligh-1.5B
- **Documentation**: `docs/learning/`
- **Issues**: https://github.com/Kandil7/Baligh-1.5B/issues