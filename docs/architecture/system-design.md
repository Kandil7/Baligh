# System Architecture Design

## Overview

Baligh-1.5B v0 follows a clean architecture pattern with clear separation of concerns.

## Module Structure

src/baligh/
config.py - Configuration management (Pydantic)
constants.py - Project constants and paths
data/ - Data pipeline
  loader.py - Dataset loading from HF
  cleaner.py - Text cleaning pipeline
  mixer.py - Dataset mixing by ratios
  formatter.py - Prompt formatting for CPT/SFT
  validators.py - Data validation
  datasets.py - Dataset registry
training/ - Training infrastructure
  cpt_trainer.py - Continued pretraining
  sft_trainer.py - Supervised fine-tuning
  lora_config.py - LoRA/QLoRA configuration
  callbacks.py - Training callbacks
  metrics.py - Training metrics
  checkpoint.py - Checkpoint management
evaluation/ - Evaluation framework
  evaluator.py - Main evaluator
  metrics.py - Evaluation metrics
  benchmarks.py - Benchmark runners
  human_eval.py - Human evaluation
  reporters.py - Report generation
models/ - Model utilities
  loader.py - Model loading with quantization
  tokenizer.py - Tokenizer handling
  quantization.py - Quantization (GGUF, AWQ, GPTQ)
inference/ - Inference utilities
  generator.py - Text generation
  chat.py - Chat interface
  structured.py - Structured output
utils/ - Shared utilities
  logging.py - Loguru setup
  seeding.py - Reproducibility
  distributed.py - Distributed training
  memory.py - Memory management

## Design Principles

1. Single Responsibility: Each module has one clear purpose
2. Dependency Injection: Config passed explicitly, no global state
3. Immutable Configs: Frozen dataclasses/Pydantic models
4. Explicit Errors: Custom exception hierarchy
5. Observability: Structured logging, metrics, tracing

## Data Flow

Raw Datasets (HF) -> Loader -> Cleaner -> Mixer -> Formatter -> Trainer
                                                      down
                                              Checkpoints -> Merge -> Quantize -> Release
                                                      down
                                              Evaluator -> Benchmarks -> Reports

## Configuration

- Base: configs/base/ (model, hardware)
- CPT: configs/cpt/ (stage1, stage2, islamic)
- SFT: configs/sft/ (stage1, stage2, islamic)
- Eval: configs/eval/
- All configs loadable via Pydantic Settings with env var override
