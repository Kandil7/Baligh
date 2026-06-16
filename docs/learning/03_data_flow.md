# Data Flow: Baligh-1.5B v0

## End-to-End Pipeline Overview

```mermaid
flowchart LR
    A["HF Hub<br>(Datasets)"] --> B["Load<br>(loader.py)"]
    B --> C["Clean<br>(CleaningPipeline)"]
    C --> D["Dedup<br>(MinHash LSH)"]
    D --> E["Validate<br>(CPT/SFT checks)"]
    E --> F["Mix<br>(interleave_datasets)"]
    F --> G["Format<br>(tokenize + template)"]
    G --> H["Split<br>(train/eval)"]
    H --> I["Output<br>(saved datasets)"]
```

> Dedup (Step 3) is **optional** and applies only to web corpora.

---

## Step 1: Load

**Source**: `loader.py` with `DATASET_REGISTRY`

All datasets are registered in a central dictionary mapping dataset names to their HuggingFace identifiers and configurations.

```mermaid
flowchart TD
    A["DATASET_REGISTRY"] --> B{"Dataset size?"}
    B -->|"Small<br>(< 1B tokens)"| C["load_dataset()<br>(full in memory)"]
    B -->|"Large<br>(> 1B tokens)"| D["load_dataset(streaming=True)<br>(iterable, no full load)"]
    D --> E["arabicweb24<br>28B tokens<br>(streaming required)"]
```

- **Streaming** is used for `arabicweb24` (28B tokens) -- the full dataset exceeds memory, so it is consumed as an iterable.
- Smaller datasets (`cidar`, `evol_instruct_arabic`, `gazelle`, etc.) are loaded fully into memory.

---

## Step 2: Clean

**Source**: `CleaningPipeline` class

Applied via `Dataset.map()` with `batched=True` and `num_proc=8` for parallel processing.

```mermaid
flowchart TD
    A["Raw Text"] --> B["HTML Tag Removal"]
    B --> C["URL Removal"]
    C --> D["Arabic Normalization"]
    D --> E["Quality Filter<br>(min 50 chars)"]
    E --> F["Cleaned Text"]
```

### Arabic Normalization Rules

| Pattern | Replacement | Description |
|---------|-------------|-------------|
| Alef variants | bare Alef | alef-madda, alef-with-hamza-above/below |
| Yaa variants | bare Yaa | alef-maqsurah, yaa-with-dots-below |
| Tatweel | removed | Tatweel/kashida character stripped |
| Hamza variants | bare Hamza | hamza-on-waw, hamza-on-yaa, hamza-above/below |

### Cleaning Steps

| Order | Operation | Detail |
|-------|-----------|--------|
| 1 | HTML removal | Strip all HTML tags |
| 2 | URL removal | Remove URLs and web references |
| 3 | Arabic normalization | Alef, Yaa, Tatweel, Hamza normalization |
| 4 | Quality filter | Discard documents shorter than 50 characters |

### Execution Configuration

```python
Dataset.map(
    cleaning_fn,
    batched=True,
    num_proc=8
)
```

---

## Step 3: Dedup (Optional)

**Algorithm**: MinHash Locality-Sensitive Hashing (LSH)

Used for **web corpora only** (e.g., `arabicweb24`). Not applied to curated-quality datasets.

```mermaid
flowchart LR
    A["Documents"] --> B["Generate<br>MinHashes"]
    B --> C["LSH Bucketing"]
    C --> D{"Similarity<br>> threshold?"}
    D -->|"Yes"| E["Duplicate<br>(discard)"]
    D -->|"No"| F["Unique<br>(keep)"]
```

- Detects **near-duplicates** (not just exact matches).
- Essential for web-scraped content where boilerplate and near-identical pages are common.

---

## Step 4: Validate

Validation rules differ by task type (CPT vs SFT).

```mermaid
flowchart TD
    A["Cleaned Dataset"] --> B{"Task Type?"}
    B -->|CPT| C["Text length<br>50 - 100K chars"]
    B -->|SFT| D{"Has instruction<br>field?"}
    D -->|No| E["Discard"]
    D -->|Yes| F["Output length<br>>= 5 chars"]
    F -->|Pass| G["Valid"]
    F -->|Fail| E
    C -->|Pass| G
    C -->|Fail| E
```

| Task | Requirement | Rationale |
|------|-------------|-----------|
| CPT | Text 50-100,000 chars | Too short = low signal; too long = memory waste |
| SFT | Instruction present | Core requirement for instruction-following |
| SFT | Output >= 5 chars | Ensures meaningful response content |

---

## Step 5: Mix

**Method**: `interleave_datasets()` with `probabilities` and `first_exhausted` stopping strategy.

```mermaid
flowchart TD
    A["CPT Mix"] --> B["arabicweb24 (70%)"]
    A --> C["arabictext_large (20%)"]
    A --> D["arabic_pile (10%)"]

    E["SFT Mix"] --> F["cidar (40%)"]
    E --> G["evol_instruct_arabic (35%)"]
    E --> H["gazelle (10%)"]
    E --> I["summarization (10%)"]
    E --> J["islamic_qa (5%)"]
```

### Mixing Strategy

- **`interleave_datasets`**: Samples from each dataset according to the given probabilities.
- **`first_exhausted`**: Training stops when the first (most-used) dataset runs out.

### CPT Probabilities

| Dataset | Probability |
|---------|-------------|
| arabicweb24 | 70% |
| arabictext_large | 20% |
| arabic_pile | 10% |

### SFT Probabilities

| Dataset | Probability |
|---------|-------------|
| cidar | 40% |
| evol_instruct_arabic | 35% |
| gazelle | 10% |
| summarization | 10% |
| islamic_qa | 5% |

### Islamic Cycle

The `islamic_qa` dataset is trained in a **separate isolated cycle** rather than being interleaved into the main SFT mix. This allows:

- Dedicated fine-tuning on Islamic knowledge.
- Independent control over Islamic domain training duration.

```mermaid
flowchart LR
    A["SFT Mix<br>(main cycle)"] --> B["Train -> stage2 complete"]
    C["Islamic Cycle<br>(isolated)"] --> D["Train -> 5000 steps"]
    B --> E["Final Model"]
    D --> E
```

---

## Step 7: Split

After formatting, the dataset is split into train and eval sets.

| Task Type | Train | Eval | Rationale |
|-----------|-------|------|-----------|
| CPT | 99% | 1% | Massive data; 1% eval is sufficient |
| SFT | 98% | 2% | Smaller datasets; 2% eval for better signal |

---

## CPT vs SFT Comparison

| Aspect | CPT (Continual Pre-Training) | SFT (Supervised Fine-Tuning) |
|--------|------------------------------|------------------------------|
| **Goal** | Language modeling on raw Arabic text | Instruction-following capability |
| **Input** | Raw text | instruction + output pairs |
| **Template** | None (raw tokenize + pack) | Qwen2.5 chat template |
| **Labels** | labels = input_ids | Labels masked to output tokens only |
| **Validation** | Text 50-100K chars | Instruction present, output >= 5 chars |
| **Eval split** | 1% | 2% |
| **Datasets** | arabicweb24, arabictext_large, arabic_pile | cidar, evol_instruct_arabic, gazelle, summarization, islamic_qa |

---

## CPT Configuration Stages

The CPT phase runs in three sequential stages with different hyperparameters:

```mermaid
flowchart LR
    A["Stage 1<br>1000 steps<br>LR: 2e-4"] --> B["Stage 2<br>50000 steps<br>LR: 2e-4"] --> C["Islamic<br>5000 steps<br>LR: 1e-4"]
```

| Stage | Steps | Learning Rate | Purpose |
|-------|-------|---------------|---------|
| stage1 | 1,000 | 2e-4 | Quick warm-up / sanity check |
| stage2 | 50,000 | 2e-4 | Main continual pre-training |
| islamic | 5,000 | 1e-4 | Islamic domain specialization |

### Stage Details

- **Stage 1** (1,000 steps): Short warm-up run to validate the pipeline end-to-end before committing to the full training run.
- **Stage 2** (50,000 steps): The main CPT phase where the bulk of language knowledge is acquired from the mixed web and text corpora.
- **Islamic** (5,000 steps): A dedicated phase at a lower learning rate (1e-4) for fine-tuning on Islamic domain data. Runs as an isolated cycle after the main SFT is complete.