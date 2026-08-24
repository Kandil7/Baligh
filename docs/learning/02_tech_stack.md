# 02 — Tech Stack: Baligh-1.7B v0

## Core ML Framework

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Base model | `transformers` | 4.40+ | Model loading, training, inference |
| LoRA adapters | `peft` | 0.10+ | QLoRA adapter injection and management |
| SFT training | `trl` | 0.8+ | SFTTrainer with response-only loss |
| Quantized loading | `bitsandbytes` | 0.43+ | 4-bit NF4 / 8-bit quantization |
| Training toolkit | `unsloth` | 2024+ | Faster QLoRA with memory optimization |
| Deep learning | `torch` | 2.2+ | Autograd, CUDA, model computation |

## Data Pipeline

| Component | Library | Purpose |
|-----------|---------|---------|
| Dataset loading | `datasets` (Hugging Face) | Load, stream, split, interleave datasets |
| Data mixing | `datasets.interleave_datasets` | Ratio-based dataset interleaving |
| Tokenization | `transformers.AutoTokenizer` | Qwen2.5 tokenizer with chat template |
| Cleaning | Custom (`cleaner.py`) | HTML removal, Arabic normalization, quality filters |
| Deduplication | MinHash LSH | Near-duplicate detection on Arabic text |

## Evaluation

| Component | Library | Purpose |
|-----------|---------|---------|
| ROUGE scores | `rouge-score` | n-gram overlap for generation quality |
| BLEU scores | `sacrebleu` | Corpus-level translation quality |
| BERTScore | `bert-score` | Semantic similarity for Arabic (lang="ar") |
| Perplexity | Custom (`metrics.py`) | Language modeling quality |

## Quantization

| Format | Library | Use Case |
|--------|---------|----------|
| GGUF | `llama-cpp-python` / `llama.cpp` | llama.cpp, Ollama, LM Studio |
| AWQ | `autoawq` | vLLM, TensorRT-LLM |
| GPTQ | `auto-gptq` | text-generation-inference, HF TGI |

## Configuration & Infrastructure

| Component | Library | Purpose |
|-----------|---------|---------|
| Config management | `pydantic` + `pydantic-settings` | Typed configs with env var support |
| Logging | `loguru` | Structured logging (JSON/text, rotation) |
| Experiment tracking | `wandb` + `tensorboard` | Loss curves, metrics, checkpoints |
| Reproducibility | `torch`, `numpy`, `random` | Deterministic seeding |
| Package management | `pyproject.toml` | Modern Python packaging |
| Containerization | `docker` | Reproducible training environments |
| CI/CD | GitHub Actions | Automated testing, training, releases |

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Qwen2.5 (decoder-only transformer) |
| Parameters | 1.54B |
| Hidden size | 2048 |
| Num layers | 28 |
| Attention heads | GQA (Grouped Query Attention) |
| Vocab size | 151,936 |
| Max context | 32,768 tokens (trained at 2048, extendable) |
| Attention impl | Flash Attention 2 (default) |
| Precision | bfloat16 compute, NF4 quantized weights |

## LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 16 | Balance between expressiveness and parameter count |
| `lora_alpha` | 16 | Scaling factor (alpha/r = 1.0) |
| `lora_dropout` | 0.0 | No dropout (Unsloth optimization) |
| `target_modules` | q, k, v, o, gate, up, down | All attention + MLP projections |
| `bias` | "none" | No bias terms in LoRA |
| `task_type` | CAUSAL_LM | Causal language modeling |

## Training Hyperparameters

### CPT (Continued Pretraining)

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 |
| Batch size | 2 per device |
| Gradient accumulation | 4 (effective batch = 8) |
| Max steps | 50,000 |
| Warmup steps | 1,000 |
| LR scheduler | Cosine |
| Optimizer | AdamW (fused) |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Mixed precision | bf16 |
| Packing | True (multiple sequences per sample) |

### SFT (Supervised Fine-Tuning)

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (lower than CPT) |
| Batch size | 2 per device |
| Gradient accumulation | 4 (effective batch = 8) |
| Max steps | 10,000 |
| Warmup steps | 500 |
| LR scheduler | Cosine |
| Optimizer | AdamW (fused) |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Mixed precision | bf16 |
| Packing | False (preserve conversation structure) |
| Response-only loss | True (mask user tokens) |

## Dataset Mix Ratios

### CPT Datasets

| Dataset | Ratio | Tokens | Domain |
|---------|-------|--------|--------|
| ArabicWeb24 | 70% | ~28B | Web crawl |
| ArabicText-Large | 20% | ~1B | General text |
| The Arabic Pile | 10% | ~5B | Mixed dialects |

### CPT Islamic Cycle (separate training pass)

| Dataset | Domain |
|---------|--------|
| hadith_datasets | Hadith reports |
| quran_qa | Quran Q&A |
| quran_md | Quran text |
| arabic_islamic_texts | General Islamic texts |

### SFT Datasets

| Dataset | Ratio | Domain |
|---------|-------|--------|
| CIDAR | 40% | Arabic instruction-following |
| evol-instruct-arabic | 35% | Evolved instructions |
| Gazelle | 10% | Arabic writing |
| Summarization | 10% | Text summarization |
| Islamic QA | 5% | Islamic question-answering |

## System Requirements

### Minimum (Inference)

- GPU: 4 GB VRAM (quantized GGUF Q4)
- RAM: 8 GB
- Storage: 2 GB (model + tokenizer)

### Recommended (Training)

- GPU: 24 GB VRAM (RTX 4090, A5000, A100 40GB)
- RAM: 32 GB
- Storage: 100 GB (datasets + checkpoints)
- CUDA: 11.8+
- Python: 3.10+

### Optimal (Full Pipeline)

- GPU: 2x A100 80GB or 4x A100 40GB
- RAM: 64 GB
- Storage: 500 GB NVMe SSD
- CUDA: 12.1+
- Python: 3.11+

## Dependencies by Concern

```
requirements/
├── base.txt          # transformers, torch, pydantic, loguru
├── training.txt      # peft, trl, bitsandbytes, unsloth, wandb
├── eval.txt          # rouge-score, sacrebleu, bert-score
├── dev.txt           # pytest, ruff, mypy, pre-commit
└── deployment.txt    # autoawq, auto-gptq, llama-cpp-python
```

## Why These Choices

| Decision | Reason |
|----------|--------|
| Qwen3-1.7B base | Best open Arabic support at 1.5B scale, GQA efficiency |
| QLoRA over full fine-tune | 75% VRAM reduction, comparable quality |
| TRL SFTTrainer | Built-in response-only loss, packing, chat template support |
| Loguru over stdlib logging | Cleaner API, built-in rotation, JSON serialization |
| Pydantic BaseSettings | Type-safe configs with env var override |
| Frozen dataclasses | Immutability prevents config corruption during training |
| Flash Attention 2 | 2-3x memory reduction for long sequences |
| NF4 over FP4 | Better quantization quality (Normal Float vs standard Float4) |
| Cosine LR schedule | Smooth decay prevents late-training instability |
| Fused AdamW | 15-20% faster optimizer step on CUDA |
