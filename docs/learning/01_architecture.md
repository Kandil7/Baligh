# 01 — Architecture: Baligh-1.7B v0

## High-Level Architecture

```mermaid
flowchart TB
    subgraph "Data Pipeline"
        A1[HF Hub Datasets] --> A2[Loader]
        A2 --> A3[Cleaner]
        A3 --> A4[Deduplicator]
        A4 --> A5[Validator]
        A5 --> A6[Mixer]
        A6 --> A7[Formatter]
        A7 --> A8[DatasetDict]
    end

    subgraph "Training Pipeline"
        B1[Base Model<br/>Qwen3-1.7B] --> B2[4-bit Quantization<br/>NF4 + Double Quant]
        B2 --> B3[LoRA Adapters<br/>r=16, alpha=16]
        B3 --> B4{Training Stage}
        B4 -->|CPT| B5[CPT Trainer<br/>HF Trainer]
        B4 -->|SFT| B6[SFT Trainer<br/>TRL SFTTrainer]
        B5 --> B7[LoRA Checkpoint]
        B6 --> B7
    end

    subgraph "Evaluation Pipeline"
        C1[Trained Model] --> C2[Evaluator]
        C2 --> C3[Benchmarks<br/>MMLU, CIDAR, Islamic]
        C2 --> C4[Metrics<br/>ROUGE, BLEU, BERTScore]
        C2 --> C5[Human Eval<br/>1-5 Rubric]
        C3 --> C6[Report Generator]
        C4 --> C6
        C5 --> C6
        C6 --> C7[JSON/Markdown Reports]
    end

    subgraph "Deployment Pipeline"
        D1[Merged Model] --> D2[Quantizer]
        D2 --> D3[GGUF / AWQ / GPTQ]
        D3 --> D4[HuggingFace Hub]
        D3 --> D5[Local Inference<br/>llama.cpp / Ollama]
        D3 --> D6[API Serving<br/>vLLM / TGI]
    end

    A8 --> B5
    A8 --> B6
    B7 --> D1
    C1 -.->|eval only| C2

    style A1 fill:#e1f5fe
    style B1 fill:#fff3e0
    style C1 fill:#e8f5e9
    style D1 fill:#fce4ec
```

---

## Module Architecture

```mermaid
flowchart LR
    subgraph "src/baligh/"
        subgraph "config.py"
            C1[BaseConfig<br/>Pydantic Settings]
            C2[ModelConfig<br/>Frozen Dataclass]
            C3[LoRAConfig<br/>Frozen Dataclass]
            C4[CPTConfig<br/>Frozen Dataclass]
            C5[SFTConfig<br/>Frozen Dataclass]
            C6[EvalConfig<br/>Frozen Dataclass]
            C7[QuantizationConfig<br/>Frozen Dataclass]
        end

        subgraph "data/"
            D1[loader.py<br/>Dataset Loading]
            D2[cleaner.py<br/>Text Cleaning]
            D3[mixer.py<br/>Dataset Mixing]
            D4[formatter.py<br/>Tokenization]
            D5[validators.py<br/>Schema Validation]
            D6[datasets.py<br/>Dataset Registry]
        end

        subgraph "training/"
            T1[cpt_trainer.py<br/>CPT Loop]
            T2[sft_trainer.py<br/>SFT Loop]
            T3[lora_config.py<br/>LoRA/Quantization]
            T4[callbacks.py<br/>Logging/Memory/Checkpoint]
            T5[metrics.py<br/>Loss/Perplexity]
            T6[checkpoint.py<br/>Save/Load/Cleanup]
        end

        subgraph "models/"
            M1[loader.py<br/>Load/Merge/Info]
            M2[tokenizer.py<br/>Chat Template]
            M3[quantization.py<br/>GGUF/AWQ/GPTQ]
        end

        subgraph "evaluation/"
            E1[evaluator.py<br/>Generate + Batch Eval]
            E2[metrics.py<br/>ROUGE/BLEU/BERTScore]
            E3[benchmarks.py<br/>MMLU/CIDAR/Islamic]
            E4[human_eval.py<br/>Rubric Scoring]
            E5[reporters.py<br/>JSON/MD Reports]
        end

        subgraph "inference/"
            I1[generator.py<br/>Text Generation]
            I2[chat.py<br/>ChatBot Interface]
            I3[structured.py<br/>JSON Extraction]
        end

        subgraph "utils/"
            U1[logging.py<br/>Loguru Setup]
            U2[seeding.py<br/>Reproducibility]
            U3[distributed.py<br/>DDP Helpers]
            U4[memory.py<br/>GPU Memory]
        end
    end

    C1 --> D1
    C2 --> M1
    C3 --> T3
    C4 --> T1
    C5 --> T2
    C6 --> E1
    D1 --> D2 --> D3 --> D4 --> D5
    T1 --> M1
    T2 --> M1
    M1 --> E1
    M2 --> D4
    M2 --> I2
    M3 --> D4

    style C1 fill:#e3f2fd
    style D1 fill:#e8f5e9
    style T1 fill:#fff3e0
    style M1 fill:#fce4ec
    style E1 fill:#f3e5f5
    style I1 fill:#e0f7fa
    style U1 fill:#f5f5f5
```

---

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "External"
        H1[ArabicWeb24<br/>28B tokens]
        H2[ArabicText-Large<br/>1B tokens]
        H3[Arabic Pile<br/>5B tokens]
        H4[CIDAR]
        H5[evol-instruct-arabic]
        H6[Gazelle]
        H7[Summarization]
        H8[Islamic QA]
        H9[Hadith, Quran]
    end

    subgraph "prepare_data.py"
        P1[Load from HF Hub<br/>streaming=True]
        P2[Clean<br/>HTML/URL/Arabic norm]
        P3[Deduplicate<br/>MinHash LSH]
        P4[Validate<br/>Schema + Quality]
        P5[Mix by Ratio<br/>interleave_datasets]
        P6[Format<br/>Tokenize + Pack]
        P7[Split<br/>Train/Eval]
    end

    subgraph "Output"
        O1[data/train_ready/cpt/<br/>DatasetDict]
        O2[data/train_ready/sft/<br/>DatasetDict]
        O3[data/train_ready/eval/<br/>DatasetDict]
    end

    H1 & H2 & H3 --> P1
    H9 --> P1
    H4 & H5 & H6 & H7 & H8 --> P1

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7

    P7 -->|CPT| O1
    P7 -->|SFT| O2
    P7 -->|Eval| O3

    style H1 fill:#bbdefb
    style H4 fill:#c8e6c9
    style H9 fill:#ffe0b2
    style O1 fill:#e1f5fe
    style O2 fill:#e8f5e9
    style O3 fill:#fff3e0
```

---

## Training Flow Architecture

```mermaid
flowchart TD
    subgraph "Stage 1: CPT"
        C1[Load Qwen3-1.7B<br/>4-bit NF4] --> C2[Apply LoRA<br/>r=16, alpha=16]
        C2 --> C3[Prepare for<br/>k-bit Training]
        C3 --> C4[CPT Data<br/>50K steps]
        C4 --> C5[Train<br/>AdamW, Cosine LR]
        C5 --> C6[Save Checkpoint]
        C6 --> C7[Load Best<br/>by eval_loss]
    end

    subgraph "Stage 2: Islamic CPT"
        I1[Load CPT Checkpoint] --> I2[Islamic Data<br/>5K steps]
        I2 --> I3[Train<br/>LR=1e-4]
        I3 --> I4[Save Checkpoint]
    end

    subgraph "Stage 3: SFT"
        S1[Load CPT+Islamic<br/>Checkpoint] --> S2[Apply LoRA<br/>r=16, alpha=16]
        S2 --> S3[SFT Data<br/>10K steps]
        S3 --> S4[Train<br/>TRL SFTTrainer<br/>Response-only Loss]
        S4 --> S5[Save Checkpoint]
        S5 --> S6[Load Best<br/>by eval_loss]
    end

    subgraph "Stage 4: SFT Stage 2"
        S7[Load SFT Checkpoint] --> S8[Hard Cases<br/>Islamic QA 15%]
        S8 --> S9[Train<br/>LR=5e-5]
        S9 --> S10[Final SFT]
    end

    C7 --> I1
    I4 --> S1
    S6 --> S7
    S10 --> M1[Merge LoRA<br/>into Base Weights]
    M1 --> M2[Quantize<br/>GGUF/AWQ/GPTQ]
    M2 --> M3[Push to HF Hub]

    style C1 fill:#e3f2fd
    style I1 fill:#fff3e0
    style S1 fill:#e8f5e9
    style M1 fill:#fce4ec
```

---

## Evaluation Flow

```mermaid
flowchart TD
    subgraph "Evaluator Setup"
        E1[Load Model<br/>4-bit] --> E2[Load LoRA<br/>Adapter]
        E2 --> E3[Set eval mode<br/>model.eval]
    end

    subgraph "Benchmarks"
        B1[MMLU-Arabic<br/>MCQ Accuracy]
        B2[CIDAR<br/>Instruction Quality]
        B3[mr-tydi Arabic<br/>Retrieval QA]
        B4[Islamic QA<br/>Knowledge Accuracy]
        B5[Perplexity<br/>Language Quality]
    end

    subgraph "Metrics"
        M1[ROUGE-1/2/L]
        M2[corpus BLEU]
        M3[BERTScore<br/>Arabic]
        M4[Exact Match]
        M5[Accuracy]
    end

    subgraph "Human Eval"
        H1[100 Samples]
        H2[5-Dimension Rubric]
        H3[CSV Export]
    end

    subgraph "Reporting"
        R1[JSON Report<br/>eval_report_*.json]
        R2[Markdown Summary]
        R3[WandB Upload]
    end

    E3 --> B1 & B2 & B3 & B4 & B5
    B1 --> M5
    B2 --> M1 & M2
    B3 --> M1 & M4
    B4 --> M1 & M4
    B5 --> M5
    E3 --> H1 --> H2 --> H3
    B1 & B2 & B3 & B4 & B5 --> R1
    R1 --> R2
    R1 --> R3

    style E1 fill:#e3f2fd
    style B1 fill:#e8f5e9
    style M1 fill:#fff3e0
    style H1 fill:#fce4ec
    style R1 fill:#f3e5f5
```

---

## Package Dependency Graph

```mermaid
flowchart TD
    config[config.py] --> constants[constants.py]
    
    data[data/] --> config
    data --> models[models/]
    data --> utils[utils/]
    
    training[training/] --> config
    training --> models
    training --> data
    training --> utils
    
    models[models/] --> config
    models --> utils
    
    evaluation[evaluation/] --> config
    evaluation --> models
    evaluation --> data
    evaluation --> utils
    
    inference[inference/] --> config
    inference --> models
    inference --> utils
    
    scripts[scripts/] --> data
    scripts --> training
    scripts --> evaluation
    scripts --> models
    scripts --> utils

    style config fill:#e3f2fd
    style utils fill:#f5f5f5
    style scripts fill:#fff3e0
```

---

## Key Interfaces

### Model Loading Interface
```python
# Loading
model = load_base_model(model_name, load_in_4bit=True)
model = apply_lora(model, lora_config)
model = prepare_model_for_kbit_training(model)

# Inference
generator = TextGenerator(model_path, adapter_path)
response = generator.generate(prompt, max_new_tokens=512)

# Chat
bot = ChatBot(model_path, adapter_path, system_prompt)
response = bot.chat(user_message)

# Merge & Export
merged = merge_lora(model, save_path)
quantize_gguf(model_path, output_path, quantization)
```

### Data Pipeline Interface
```python
# Loading
datasets = load_cpt_datasets(streaming=True)
datasets = load_sft_datasets()

# Processing
cleaned = cleaning_pipeline.clean_batch(texts)
mixed = mix_cpt_datasets(datasets, seed=42)
formatted = mixed.map(formatter, batched=True)

# Validation
valid, invalid, errors = validate_dataset(ds, dataset_type="cpt")
```

### Evaluation Interface
```python
# Setup
evaluator = Evaluator(model_path, adapter_path)

# Benchmarks
results = run_mmlu_arabic(evaluator, max_samples=100)
results = run_cidar_eval(evaluator, max_samples=100)
results = run_islamic_qa(evaluator, dataset, max_samples=100)

# Metrics
rouge = compute_rouge(predictions, references)
bleu = compute_bleu(predictions, references)
bert = compute_bert_score(predictions, references, lang="ar")

# Reporting
generate_eval_report(results, output_dir)
```

---

## Configuration Hierarchy

```mermaid
flowchart TD
    subgraph "Environment"
        ENV[.env file<br/>HF_TOKEN, WANDB_API_KEY]
    end

    subgraph "YAML Configs"
        Y1[configs/base/model.yaml]
        Y2[configs/base/hardware.yaml]
        Y3[configs/cpt/cpt-stage1.yaml]
        Y4[configs/cpt/cpt-stage2.yaml]
        Y5[configs/cpt/cpt-islamic.yaml]
        Y6[configs/sft/sft-stage1.yaml]
        Y7[configs/sft/sft-stage2.yaml]
        Y8[configs/eval/eval-config.yaml]
    end

    subgraph "Python Configs"
        P1[BaseConfig<br/>Pydantic Settings]
        P2[ModelConfig<br/>Dataclass]
        P3[LoRAConfig<br/>Dataclass]
        P4[CPTConfig<br/>Dataclass]
        P5[SFTConfig<br/>Dataclass]
        P6[EvalConfig<br/>Dataclass]
    end

    subgraph "CLI Args"
        C1[--data-dir]
        C2[--output-dir]
        C3[--config]
        C4[--resume]
        C5[--max-samples]
    end

    ENV --> P1
    Y1 --> P2
    Y3 --> P4
    Y6 --> P5
    Y8 --> P6
    C1 & C2 & C3 & C4 & C5 -->|override| P4 & P5

    style ENV fill:#ffe0b2
    style Y1 fill:#e3f2fd
    style P1 fill:#e8f5e9
    style C1 fill:#fce4ec
```

**Priority order**: CLI args > YAML config > Python defaults > Environment variables
