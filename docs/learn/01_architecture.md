# Architecture - Baligh-1.5B v0

## System Architecture Overview

```mermaid
graph TB
    subgraph "Configuration Layer"
        A1[BaseConfig] --> A2[ModelConfig]
        A1 --> A3[CPTConfig]
        A1 --> A4[SFTConfig]
        A1 --> A5[EvalConfig]
        A1 --> A6[LoRAConfig]
        A1 --> A7[QuantizationConfig]
    end

    subgraph "Data Layer"
        B1[Dataset Registry] --> B2[Loader]
        B2 --> B3[Cleaner]
        B3 --> B4[Mixer]
        B4 --> B5[Formatter]
        B5 --> B6[Validator]
    end

    subgraph "Model Layer"
        C1[Loader] --> C2[QLoRA Applicator]
        C2 --> C3[Merger]
        C3 --> C4[Quantizer]
    end

    subgraph "Training Layer"
        D1[CPT Trainer] --> D2[TRL Trainer]
        D3[SFT Trainer] --> D4[TRL SFTTrainer]
        D2 --> D5[Callbacks]
        D4 --> D5
    end

    subgraph "Evaluation Layer"
        E1[Evaluator] --> E2[Benchmarks]
        E1 --> E3[Metrics]
        E1 --> E4[Human Eval]
        E2 --> E5[Reporters]
    end

    subgraph "Inference Layer"
        F1[TextGenerator] --> F2[ChatBot]
        F1 --> F3[StructuredOutput]
    end

    subgraph "Utility Layer"
        G1[Logging] --> G2[Seeding]
        G2 --> G3[Memory]
        G3 --> G4[Distributed]
    end

    B6 --> D1
    B6 --> D3
    C2 --> D1
    C2 --> D3
    D1 --> C3
    D3 --> C3
    C4 --> F1
    C4 --> E1
```

---

## Package Structure

```
src/baligh/
├── __init__.py
├── config.py              # Pydantic settings + frozen dataclasses
├── constants.py           # Paths, model names, dataset ratios
├── data/                  # Data pipeline (7 files)
│   ├── __init__.py        # Public API re-exports
│   ├── datasets.py        # Dataset registry with metadata
│   ├── loader.py          # HF dataset loading by name
│   ├── cleaner.py         # HTML/URL removal, Arabic normalization
│   ├── mixer.py           # Ratio-based interleaving
│   ├── formatter.py       # CPT packing + SFT chat template
│   └── validators.py      # Schema + quality validation
├── training/              # Training loops (7 files)
│   ├── __init__.py        # Public API re-exports
│   ├── cpt_trainer.py     # CPT training with HF Trainer
│   ├── sft_trainer.py     # SFT training with TRL SFTTrainer
│   ├── lora_config.py     # LoRA + BitsAndBytes config creation
│   ├── callbacks.py       # Logging, memory, checkpoint callbacks
│   ├── metrics.py         # Loss, perplexity computation
│   └── checkpoint.py      # Save/load/cleanup checkpoints
├── models/                # Model management (4 files)
│   ├── __init__.py        # Public API re-exports
│   ├── loader.py          # load_base_model, apply_lora, merge_lora
│   ├── tokenizer.py       # Qwen2.5 tokenizer + chat template
│   └── quantization.py    # GGUF, AWQ, GPTQ export
├── evaluation/            # Evaluation (6 files)
│   ├── __init__.py        # Public API re-exports
│   ├── evaluator.py       # Main Evaluator class
│   ├── benchmarks.py      # MMLU-Arabic, CIDAR, Islamic QA
│   ├── metrics.py         # ROUGE, BLEU, BERTScore, EM
│   ├── human_eval.py      # Human eval rubric system
│   └── reporters.py       # JSON/Markdown report generation
├── inference/             # Inference (4 files)
│   ├── __init__.py        # Public API re-exports
│   ├── generator.py       # TextGenerator (single/batch)
│   ├── chat.py            # ChatBot (history + system prompts)
│   └── structured.py      # JSON extraction with retries
└── utils/                 # Utilities (5 files)
    ├── __init__.py
    ├── logging.py         # Loguru setup (JSON/text, rotation)
    ├── seeding.py         # Deterministic seeding across all libs
    ├── memory.py          # GPU memory tracking + estimation
    └── distributed.py     # DDP helpers (barrier, reduce, gather)
```

---

## Data Flow Architecture

```mermaid
graph LR
    subgraph "Raw Data"
        R1[ArabicWeb24 - 28B tokens]
        R2[ArabicText-Large - 1B tokens]
        R3[Arabic Pile - 5B tokens]
        R4[Hadith/Quran datasets]
        R5[CIDAR - instruction pairs]
        R6[evol-instruct-arabic]
        R7[Gazelle - writing]
        R8[Summarization]
        R9[Islamic QA]
    end

    subgraph "CPT Pipeline"
        C1[Load] --> C2[Clean]
        C2 --> C3[Deduplicate]
        C3 --> C4[Validate]
        C4 --> C5[Mix 70/20/10]
        C5 --> C6[Tokenize + Pack]
    end

    subgraph "SFT Pipeline"
        S1[Load] --> S2[Clean]
        S2 --> S3[Validate]
        S3 --> S4[Mix 40/35/10/10/5]
        S4 --> S5[Apply Chat Template]
    end

    R1 & R2 & R3 --> C1
    R4 --> C1
    R5 & R6 & R7 & R8 & R9 --> S1
    C6 --> T1[CPT Training Data]
    S5 --> T2[SFT Training Data]
```

---

## Training Pipeline

```mermaid
graph TD
    subgraph "Stage 1: CPT"
        A1[Load Base Model 4-bit] --> A2[Apply LoRA r=16]
        A2 --> A3[Prepare for k-bit Training]
        A3 --> A4[Create TrainingArguments]
        A4 --> A5[HF Trainer]
        A5 --> A6[CPT Training Loop]
        A6 --> A7[Save Checkpoints]
        A7 --> A8[Final CPT Model]
    end

    subgraph "Stage 2: SFT"
        B1[Load CPT Model 4-bit] --> B2[Apply LoRA r=16]
        B2 --> B3[Prepare for k-bit Training]
        B3 --> B4[Create SFTConfig]
        B4 --> B5[TRL SFTTrainer]
        B5 --> B6[SFT Training Loop]
        B6 --> B7[Save Checkpoints]
        B7 --> B8[Final SFT Model]
    end

    A8 --> B1
```

---

## Evaluation Pipeline

```mermaid
graph TD
    E1[Load Model + Adapter] --> E2[Evaluator]
    E2 --> E3[MMLU-Arabic]
    E2 --> E4[CIDAR Eval]
    E2 --> E5[CIDAR MCQ]
    E2 --> E6[mr-tydi Arabic]
    E2 --> E7[Islamic QA]
    E3 --> E8[Compute Metrics]
    E4 --> E8
    E5 --> E8
    E6 --> E8
    E7 --> E8
    E8 --> E9[Generate Report]
    E9 --> E10[JSON + Markdown]
```

---

## Deployment Pipeline

```mermaid
graph LR
    D1[Merged FP16 Model] --> D2[GGUF q4_k_m]
    D1 --> D3[AWQ 4-bit]
    D1 --> D4[GPTQ 4-bit]
    D2 --> D5[llama.cpp / Ollama]
    D3 --> D6[vLLM / TGI]
    D4 --> D7[AutoGPTQ]
    D2 --> D8[Push to HF Hub]
    D3 --> D8
    D4 --> D8
```

---

## Key Design Patterns

### 1. Registry Pattern
**Location**: `src/baligh/data/datasets.py`

The `DATASETS` dictionary maps string names to configuration dictionaries. Each entry contains the HF path, split, streaming flag, column names, license, token count, domain, and type.

### 2. Factory Pattern
**Location**: `src/baligh/config.py` (factory functions), `src/baligh/data/loader.py`, `src/baligh/models/loader.py`

Factory functions create configured instances without exposing construction details.

### 3. Strategy Pattern
**Location**: `src/baligh/data/formatter.py`

`PromptFormatter.__call__()` dispatches based on the batch schema:
- If batch contains `instruction` key: use SFT formatting (chat template)
- If batch contains `text` key: use CPT formatting (raw tokenization)

### 4. Template Method Pattern
**Location**: `src/baligh/training/cpt_trainer.py`, `src/baligh/training/sft_trainer.py`

Both `CPTTrainer` and `SFTTrainer` follow the same lifecycle:
1. `__init__` - Setup config, tokenizer, model, training args, trainer
2. `_setup_model` - Load base model, apply LoRA, prepare for k-bit
3. `_create_training_args` - Build TrainingArguments from config
4. `_create_trainer` - Build Trainer/SFTTrainer instance
5. `train()` - Run training, save model
6. `save_model()` - Save to disk

### 5. Pipeline Pattern
**Location**: `src/baligh/data/cleaner.py`

The `CleaningPipeline` chains multiple text transformations.

### 6. Callback Pattern
**Location**: `src/baligh/training/callbacks.py`

Three `TrainerCallback` subclasses hook into training events.

### 7. Facade Pattern
**Location**: `src/baligh/data/__init__.py`, `src/baligh/models/__init__.py`

Each package's `__init__.py` re-exports a curated public API.

---

## Configuration Hierarchy

```mermaid
graph TD
    subgraph "BaseConfig (Pydantic BaseSettings)"
        BC1[project_name]
        BC2[version]
        BC3[environment]
        BC4[data_dir, config_dir, output_dir, logs_dir]
        BC5[device, mixed_precision, gradient_checkpointing]
        BC6[world_size, local_rank, ddp_backend]
        BC7[log_level, log_format, wandb_*]
        BC8[seed, deterministic]
    end

    subgraph "ModelConfig (frozen dataclass)"
        MC1[model_name: Qwen/Qwen2.5-1.5B]
        MC2[unsloth_model_name: unsloth/...]
        MC3[max_seq_length: 2048]
        MC4[load_in_4bit: True]
        MC5[bnb_4bit_quant_type: nf4]
        MC6[attn_implementation: flash_attention_2]
    end

    subgraph "LoRAConfig (frozen dataclass)"
        LC1[r: 16]
        LC2[lora_alpha: 16]
        LC3[target_modules: 7 modules]
    end

    subgraph "CPTConfig (frozen dataclass)"
        CC1[dataset_mix: 70/20/10]
        CC2[max_steps: 50000]
        CC3[learning_rate: 2e-4]
        CC4[packing: True]
    end

    subgraph "SFTConfig (frozen dataclass)"
        SC1[dataset_mix: 40/35/10/10/5]
        SC2[max_steps: 10000]
        SC3[learning_rate: 1e-4]
        SC4[packing: False]
        SC5[response_only_loss: True]
    end
```
