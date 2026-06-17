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

## Release Artifacts

- **Base model**: After CPT
- **Instruct model**: After SFT
- **Quantized variants**: GGUF, AWQ, GPTQ

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
├── tests/                # Test suite
├── docs/                 # Documentation
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
