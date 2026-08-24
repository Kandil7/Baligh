# Baligh-1.7B v0 - Complete Project Learning Documentation

This document contains the comprehensive learning documentation for the Baligh-1.7B v0 project, extracted from an OpenCode session and organized for reference.

---

## Table of Contents

1. [Key Concepts Glossary](#key-concepts-glossary)
2. [Engineering Decisions](#engineering-decisions)
3. [Code Quality Standards](#code-quality-standards)
4. [Gotchas and Tips](#gotchas-and-tips)

---

# Key Concepts Glossary

## Arabic NLP Terms

| Term | Arabic | Definition |
|------|--------|------------|
| **Fusha** | الفصحى | Modern Standard Arabic. Standardized, literary Arabic used in media, education, formal writing across Arab world |
| **Tashkeel** | التشكيل | Diacritical marks (harakat). Short vowel marks (fatha, damma, kasra, sukun) placed on consonants; often stripped in NLP |
| **Tatweel** | ـ | Elongation character. Visual stretching of Arabic words; no semantic meaning; removed during normalization |
| **Arabic Normalization** | التجويد العربي | Canonicalizing character variants: أ/إ/آ → ا, ى → ي, removal of ZWNJ/ZWJ |
| **MinHash LSH** | التجزئة المحلية المتجهة | Locality-Sensitive Hashing with MinHash for approximate near-duplicate detection |
| **Shingle (k-gram)** | النطاق | Contiguous sequence of k items (words/chars); used for Jaccard similarity estimation |

---

## Islamic Knowledge Terms

| Term | Arabic | Definition |
|------|--------|------------|
| **Fiqh** | فقه | Islamic jurisprudence. Human understanding/application of Sharia; derived from Quran, Hadith, consensus, analogy |
| **Tafsir** | تفسير | Quranic exegesis. Interpretation/explanation of Quranic verses; linguistic, legal, theological |
| **Hadith** | حديث | Reports of Prophet's sayings/actions. Secondary source of Islamic law; graded by authenticity (sahih, hasan, da'if) |
| **Aqidah** | عقيدة | Islamic creed/theology. Core beliefs: Tawhid, angels, books, prophets, Day of Judgment, Qadar |
| **Fatwa** | فتوى | Legal opinion. Non-binding legal ruling by qualified scholar (mufti); this model is NOT an authority |
| **Sharia** | الشريعة | Islamic law. Divine law from Quran/Sunnah; Fiqh is human understanding of Sharia |
| **Sunnah** | السنة | Prophetic tradition. Practices/habits of Prophet Muhammad; second source after Quran |
| **Sahih** | صحيح | Authentic. Hadith grading: sound chain, reliable narrators, no defects |
| **Isnad** | الإسناد | Chain of narrators. Chain of transmission for a Hadith; critical for authenticity grading |

---

## ML Training Terms

| Term | Definition |
|------|------------|
| **CPT (Continued Pretraining)** | Domain adaptation on raw text corpora; model learns language patterns, vocabulary, domain knowledge |
| **SFT (Supervised Fine-Tuning)** | Instruction tuning on (prompt, response) pairs; teaches instruction following, style, formatting |
| **QLoRA (Quantized LoRA)** | 4-bit quantized base model + Low-Rank Adapters; freezes base weights, trains ~1-2% params |
| **LoRA (Low-Rank Adaptation)** | Decomposes weight update ΔW = A×B where A∈R^(d×r), B∈R^(r×d); r << d |
| **Unsloth** | Training toolkit with fused kernels for 2x faster QLoRA; optimized Triton kernels |
| **NF4 (NormalFloat4)** | 4-bit quantization optimized for normally distributed weights; better than uniform INT4 |
| **Double Quantization** | Quantizes quantization constants (scales); saves additional 0.4 bits/parameter |
| **Response-only Loss** | Loss computed only on assistant tokens; user tokens masked with -100 |
| **Packing** | Concatenating multiple sequences to fill `max_seq_length`; improves GPU utilization |
| **Gradient Checkpointing** | Recompute activations during backward pass; trades compute for memory |
| **Gradient Accumulation** | Accumulate gradients over N steps before optimizer step; simulates larger batch |
| **Flash Attention 2** | Memory-efficient attention kernel; O(N²) → O(N) memory via tiling |
| **AdamW Fused** | Fused AdamW kernel (single kernel for step); 20-30% faster |
| **bfloat16** | Brain Float 16; same exponent range as FP32, 7-bit mantissa; no gradient scaling needed |

---

## Training Hyperparameters

| Term | Baligh Value | Explanation |
|------|--------------|-------------|
| **LoRA Rank (r)** | 16 | Rank of adaptation matrices; 16 = 16×d + d×16 params per layer |
| **LoRA Alpha (α)** | 16 | Scaling factor; effective scale = α/r = 1.0 |
| **LoRA Dropout** | 0.0 | No dropout on LoRA; base model frozen provides regularization |
| **Target Modules** | All attention + MLP | q/k/v/o_proj + gate/up/down_proj |
| **Learning Rate (CPT)** | 2e-4 | Standard for continued pretraining |
| **Learning Rate (SFT)** | 1e-4 | Lower for alignment; prevents catastrophic forgetting |
| **Warmup Steps** | 1000 (CPT) / 500 (SFT) | Linear warmup; prevents early instability |
| **LR Scheduler** | Cosine | Smooth decay to 0; better final convergence |
| **Weight Decay** | 0.01 | L2 regularization on weights |
| **Max Grad Norm** | 1.0 | Gradient clipping; prevents explosion |
| **Batch Size** | 2 (per device) | Fits in 8-12GB VRAM with QLoRA |
| **Grad Accum** | 4 | Effective batch = 8 |
| **Max Seq Length** | 2048 | Training context; extendable to 32K |
| **Mixed Precision** | bf16 | Bfloat16; no gradient scaling needed |

---

## Model Architecture (Qwen3-1.7B)

| Component | Specification |
|-----------|---------------|
| **Parameters** | 1.54B |
| **Layers** | 28 |
| **Hidden Size** | 2048 |
| **Attention Heads** | 16 (Grouped Query Attention - GQA) |
| **KV Heads** | 8 (GQA ratio 2:1) |
| **Hidden Act** | SwiGLU (MLP) |
| **Max Context** | 32,768 tokens |
| **Vocab Size** | 151,936 |
| **Tokenizer** | BPE (Unigram); Arabic supported natively |

---

## Quantization Formats

| Format | Use Case | Quality | Speed | Compatibility |
|--------|----------|---------|-------|---------------|
| **GGUF q4_k_m** | CPU/edge (llama.cpp) | ★★★★☆ | Fast | Universal (llama.cpp, Ollama, LM Studio) |
| **AWQ 4-bit** | GPU inference | ★★★★★ | Fastest | vLLM, TGI, TensorRT-LLM |
| **GPTQ 4-bit** | GPU inference | ★★★★☆ | Fast | AutoGPTQ, ExLlamaV2 |
| **FP16/BF16** | Training/Full precision | ★★★★★ | Baseline | Native PyTorch |

**Recommendation**: GGUF q4_k_m for distribution; AWQ for production GPU serving

---

## Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Perplexity (PPL)** | exp(avg cross-entropy loss) | Language modeling quality |
| **ROUGE-1/2/L** | N-gram overlap (recall-oriented) | Summarization, QA |
| **BLEU** | N-gram precision + brevity penalty | Translation, generation |
| **BERTScore** | Contextual embedding similarity | Semantic equivalence |
| **Exact Match** | String equality (normalized) | MCQ, closed QA |
| **Accuracy** | Correct predictions / total | MCQ benchmarks (MMLU) |

---

# Engineering Decisions

## Decision Template

Each decision follows: **Decision → Context → Options → Chosen → Trade-offs → Consequences**

---

## 1. Base Model: Qwen3-1.7B Base (not Instruct)

**Decision**: Use Qwen3-1.7B **Base** (pretrained) not Instruct

**Context**: Need Arabic-first model with Islamic specialization; starting point determines capabilities

**Options Considered**:
| Option | Pros | Cons |
|--------|------|------|
| Qwen3-1.7B Base | Clean slate; full control over Arabic adaptation | Requires full CPT + SFT |
| Qwen3-1.7B Instruct | Ready for chat; less training | Baked-in English alignment; harder to Arabic-ize |
| LLaMA-2-1.5B | - | Doesn't exist |
| Mistral-1.5B | - | Doesn't exist |
| BLOOM-1.5B | Multilingual | Weak Arabic; deprecated |

**Chosen**: **Qwen3-1.7B Base**

**Rationale**:
- Native Arabic support in tokenizer/vocab
- 32K context (vs 4K/8K in older models)
- GQA (Grouped Query Attention) - efficient inference
- Unsloth provides pre-quantized 4-bit version
- Apache 2.0 license (commercial friendly)

**Trade-offs**:
- ✅ Full control over Arabic personality
- ✅ No English-centric alignment to undo
- ❌ Requires 2-stage training (CPT + SFT) vs 1-stage
- ❌ More compute/time than Instruct fine-tune only

**Consequences**: 
- CPT stage mandatory (~8 hrs on A100)
- Total training ~12-15 hrs vs ~4 hrs for Instruct-only
- But: Far superior Arabic fluency and Islamic knowledge grounding

---

## 2. Training Method: QLoRA 4-bit (not Full Fine-Tuning)

**Decision**: QLoRA 4-bit (NF4) with LoRA r=16, α=16

**Context**: 1.5B model full fine-tuning requires ~24GB VRAM; target is 8-12GB (Colab T4 / single A100)

**Options Considered**:
| Method | VRAM (1.5B) | Quality | Speed |
|--------|-------------|---------|-------|
| Full FT (fp16) | ~24GB | Best | Baseline |
| Full FT (8-bit) | ~16GB | Good | Slower |
| LoRA (fp16) | ~12GB | Good | Fast |
| **QLoRA 4-bit (NF4)** | **8-10GB** | **Near-full** | **Fast (Unsloth)** |
| LoRA 8-bit | ~10GB | Good | Fast |

**Chosen**: **QLoRA 4-bit NF4, r=16, α=16**

**Rationale**:
- NF4 (NormalFloat4) optimized for normal weight distributions
- Double quantization saves additional 0.4 bits/param
- Unsloth provides 2x speedup via fused kernels
- r=16 sufficient for 1.5B (1-2% trainable params)

**Trade-offs**:
- ✅ Fits in 8GB (Colab T4) / 12GB (RTX 3080)
- ✅ 2x faster via Unsloth kernels
- ✅ Near-full fine-tuning quality (per QLoRA paper)
- ❌ Slightly lower ceiling than full FT
- ❌ Cannot modify base model embeddings (frozen)

**Consequences**:
- Enables free Colab training (T4 16GB)
- Enables local training on consumer GPUs
- Slight quality ceiling vs full FT (acceptable for v0)

---

## 3. Training Stages: CPT → SFT (not SFT-only)

**Decision**: Two-stage: Continued Pretraining → Supervised Fine-Tuning

**Context**: Base model needs Arabic fluency before instruction following

**Options Considered**:
| Approach | Pros | Cons |
|----------|------|------|
| **CPT → SFT** | Arabic fluency first; then instruction following | 2x training time |
| SFT only on Base | Fast (~4 hrs) | Poor Arabic fluency; hallucinates Arabic |
| CPT only | Good Arabic LM | No instruction following |

**Chosen**: **CPT → SFT**

**Rationale**:
- CPT teaches Arabic fluency, spelling, morphology, Islamic terminology
- SFT teaches instruction following, chat format, structured output
- Separation of concerns: language modeling vs alignment

**Trade-offs**:
- ✅ Strong Arabic foundation before alignment
- ✅ Modular: can swap SFT data without re-running CPT
- ❌ 2x training time (~12-15 hrs total)
- ❌ Two checkpoint management

**Consequences**:
- CPT checkpoint (`training/cpt/final/`) reusable for different SFT experiments
- SFT can iterate on data mix without re-running CPT
- Clear separation: CPT = language modeling, SFT = alignment

---

## 4. Tokenizer: Keep Qwen2.5 Original (No Modification)

**Decision**: Use Qwen2.5 tokenizer as-is; no Arabic-specific tokenizer training

**Context**: Tokenizer choice affects vocabulary, compression, and compatibility

**Options Considered**:
| Option | Pros | Cons |
|--------|------|------|
| **Keep Qwen2.5 tokenizer** | Compatible with base weights; Arabic already supported | Suboptimal compression for Arabic |
| **Train new Arabic tokenizer** | Better compression (30% fewer tokens) | Breaks weight compatibility; need full retraining |
| **Extend Qwen tokenizer** | Middle ground | Complex; still changes embeddings |

**Chosen**: **Keep Qwen2.5 tokenizer as-is**

**Rationale**:
- Qwen2.5 vocab (151,936) includes Arabic tokens
- Arabic compression ~1.3x vs English (acceptable)
- **Zero risk** to base model compatibility
- Can train custom tokenizer in v1/v2 if needed

**Trade-offs**:
- ✅ Zero risk to base model weights
- ✅ Compatible with all Qwen ecosystem tools
- ❌ ~30% more tokens for Arabic vs custom tokenizer
- ❌ Slightly higher training cost (more tokens)

**Consequences**: 
- v0 ships with standard Qwen tokenizer
- v1/v2 can explore custom tokenizer if compression critical

---

## 5. Data Mixing: `interleave_datasets` (not Concatenation)

**Decision**: Use `interleave_datasets(probabilities=...)` for ratio-based mixing

**Context**: Multiple datasets with different sizes/domains need balanced exposure

**Options Considered**:
| Method | Behavior | Problem |
|--------|----------|---------|
| **Concatenate** | All A, then all B, then all C | Domain blocks; catastrophic forgetting within epoch |
| **Interleave (round-robin)** | A, B, C, A, B, C... | Uniform but ignores dataset size differences |
| **Interleave (probabilistic)** | A, A, B, A, C, B... | Matches ratios; uniform per-step exposure |

**Chosen**: **`interleave_datasets(probabilities=ratios, stopping_strategy="first_exhausted")`**

**Rationale**:
- Matches exact ratio targets per training step
- `first_exhausted` prevents oversampling large datasets
- Seed=42 for reproducibility

**Trade-offs**:
- ✅ Exact ratio adherence per step
- ✅ Reproducible (fixed seed)
- ✅ Stops when smallest dataset exhausted (but CPT datasets are huge)
- ❌ Slightly more complex than concatenation

---

## 6. SFT: Response-only Loss (not Full Sequence)

**Decision**: Mask user tokens in loss; only compute gradients on assistant tokens

**Context**: SFT teaches instruction following; user tokens are context, not targets

**Options Considered**:
| Loss Type | Behavior | Quality |
|-----------|----------|---------|
| **Full sequence** | Loss on all tokens | Wastes capacity on user text |
| **Response-only** | Loss only on assistant | Focuses on generation |
| **Weighted** | User: 0.1x, Assistant: 1.0x | Compromise |

**Chosen**: **Response-only loss (mask user tokens with -100)**

**Rationale**:
- User tokens are context, not prediction targets
- Forces model to learn conditional generation P(response | instruction)
- Standard in InstructGPT, Alpaca, etc.

**Implementation**:
```python
labels = input_ids.copy()
assistant_start = find_assistant_token_start(input_ids)
labels[:assistant_start] = -100  # CrossEntropyLoss ignores -100
```

**Trade-offs**:
- ✅ Focuses capacity on generation
- ✅ Standard practice in SFT literature
- ❌ Slightly more complex formatting
- ❌ Requires accurate chat template parsing

---

## 7. CPT: Packing Enabled; SFT: Packing Disabled

**Decision**: `packing=True` for CPT; `packing=False` for SFT

**Context**: Packing concatenates sequences to fill `max_seq_length`

**Options Considered**:
| Stage | Packing | Rationale |
|-------|---------|-----------|
| **CPT** | **Enabled** | Raw text has no structure; packing = 3-5x throughput |
| **SFT** | **Disabled** | Conversations have structure; packing breaks turn boundaries |

**Chosen**: **CPT=True, SFT=False**

**Rationale**:
- CPT: Raw text has no semantic boundaries; packing = free throughput
- SFT: Conversations have semantic structure (user→assistant); packing mixes turns

**Trade-offs**:
| Stage | Packing | Throughput | Quality Risk |
|-------|---------|------------|--------------|
| CPT | ✅ | 3-5x tokens/sec | None (no structure) |
| SFT | ❌ | Baseline | Turn boundary corruption if enabled |

---

## 8. CPT: Cosine LR; SFT: Cosine LR (Lower Peak)

**Decision**: Cosine LR schedule for both; SFT peak LR = 1e-4 (vs CPT 2e-4)

**Context**: LR schedule affects convergence and final quality

**Options Considered**:
| Schedule | CPT | SFT |
|----------|-----|-----|
| Constant | ❌ | ❌ |
| Linear decay | ⚠️ | ⚠️ |
| **Cosine** | ✅ | ✅ |
| Cosine with restarts | Optional | ❌ |

**Chosen**: **Cosine for both**

**Rationale**:
- Cosine provides smooth decay to 0
- No sudden LR drops (unlike step decay)
- Standard in LLM training (LLaMA, PaLM, etc.)

**SFT Peak LR = 1e-4** (half of CPT's 2e-4):
- Alignment needs smaller updates
- Prevents catastrophic forgetting of CPT knowledge
- Standard in InstructGPT, Alpaca, etc.

---

## 9. Optimizer: AdamW Fused (not Standard AdamW)

**Decision**: `optim="adamw_torch_fused"` with β1=0.9, β2=0.95

**Context**: Optimizer choice affects convergence speed and stability

**Chosen**: **AdamW Fused** with β2=0.95 (higher than default 0.999)

**Rationale**:
- `adamw_torch_fused` → Single fused kernel; 20-30% faster
- β2=0.95 → Faster adaptation to gradient variance (recommended for LLMs)
- ε=1e-8 (default) - Numerical stability

**Trade-offs**:
- ✅ 20-30% faster training steps
- ✅ β2=0.95 stabilizes LLM training (per PaLM, Chinchilla papers)
- ❌ Only on CUDA 11.7+ (requires Ampere+ GPU)

---

## 10. Precision: bfloat16 (not fp16)

**Decision**: `bf16=True` (mixed precision bfloat16)

**Context**: Mixed precision training for memory/speed

**Options**:
| Precision | Gradient Scaling | Dynamic Range | Hardware |
|-----------|------------------|---------------|----------|
| **fp16** | Required | Limited (65K) | All CUDA |
| **bf16** | **Not needed** | **Same as FP32** | **Ampere+** |

**Chosen**: **bfloat16**

**Rationale**:
- No gradient scaling needed (no overflow/underflow)
- Same dynamic range as FP32
- Native on Ampere+ (A100, H100, RTX 30/40)
- Native on Google TPU v4+

**Trade-offs**:
- ✅ No gradient scaler bugs
- ✅ Stable training
- ❌ Requires Ampere+ GPU (no T4, V100 support)

**Note**: Colab T4 doesn't support bf16; Colab uses `fp16=True` fallback

---

## 11. Unsloth over Standard HF Trainer

**Decision**: Use Unsloth for QLoRA training

**Context**: Unsloth provides fused kernels for 2x QLoRA speed

**Options**:
| Trainer | Speed | Memory | Complexity |
|---------|-------|--------|------------|
| **Unsloth + TRL** | **2x** | **70%** | Medium |
| Standard HF Trainer | Baseline | Baseline | Low |
| DeepSpeed ZeRO-3 | 1.5x | 50% | High |

**Chosen**: **Unsloth + TRL SFTTrainer**

**Rationale**:
- Pre-quantized models (`unsloth/Qwen3-1.7B-Base`)
- Fused kernels for QLoRA forward/backward
- Drop-in replacement for HF Trainer

**Trade-offs**:
- ✅ 2x training speed
- ✅ 30% less VRAM
- ❌ External dependency (Git install)
- ❌ Less battle-tested than core HF

---

## 12. Evaluation: Multiple Benchmarks (not Single)

**Decision**: Multi-benchmark evaluation suite

**Context**: Single metric (perplexity) insufficient for Arabic/Islamic quality

**Chosen Benchmarks**:
| Benchmark | Type | Metric | Purpose |
|-----------|------|--------|---------|
| **MMLU-Arabic** | MCQ | Accuracy | General knowledge |
| **CIDAR-EVAL-100** | Open-ended | ROUGE-L | Cultural relevance |
| **CIDAR-MCQ-100** | MCQ | Accuracy | Cultural knowledge |
| **mr-tydi Arabic** | Retrieval QA | F1/EM | Grounded QA |
| **Islamic QA Custom** | Open-ended | ROUGE-L + EM | Domain knowledge |
| **Human Eval** | 1-5 rubric | Mean score | Subjective quality |

**Why not just perplexity?** → PPL correlates poorly with instruction following, factuality, Arabic quality

---

## 13. Release: Multiple Quantized Variants

**Decision**: Ship Base + Instruct + GGUF + AWQ + GPTQ

**Context**: Different deployment targets need different formats

**Chosen Variants**:
| Variant | Target | Size | Tool |
|---------|--------|------|------|
| **Base (merged)** | Fine-tuning base | ~3GB | `transformers` |
| **Instruct (merged)** | General inference | ~3GB | `transformers` |
| **GGUF q4_k_m** | CPU/Edge | ~1GB | `llama.cpp`, Ollama |
| **AWQ 4-bit** | GPU Server | ~1GB | vLLM, TGI |
| **GPTQ 4-bit** | GPU Legacy | ~1GB | AutoGPTQ |

**Why not just one?** → Deployment diversity; users choose based on hardware

---

# Code Quality Standards

## Testing Strategy

### Test Pyramid

```
        E2E Tests (1)
           /    \
    Integration (5)
         /      
    Unit Tests (50+)
```

### Test Organization

```
tests/
├── unit/
│   ├── test_config.py           # Config validation, defaults
│   ├── test_cleaner.py          # Arabic normalization, quality filters
│   ├── test_mixer.py            # Ratio mixing, interleave behavior
│   ├── test_formatter.py        # CPT packing, SFT chat template
│   ├── test_validators.py       # Schema validation
│   ├── test_lora_config.py      # LoRA config creation
│   └── test_utils.py            # Seeding, memory, distributed
├── integration/
│   ├── test_data_pipeline.py    # End-to-end prepare_data
│   ├── test_cpt_training.py     # CPT trainer smoke test
│   ├── test_sft_training.py     # SFT trainer smoke test
│   └── test_evaluation.py       # Evaluator + benchmarks
└── fixtures/
    ├── sample_cpt_data.jsonl
    ├── sample_sft_data.jsonl
    └── sample_eval_data.jsonl
```

---

### Test Commands

```bash
# All tests with coverage
make test
# or
pytest tests/ -v --cov=src/baligh --cov-fail-under=80

# Unit only (fast)
pytest tests/unit -v

# Integration (requires GPU)
pytest tests/integration -v --gpu

# Specific module
pytest tests/unit/test_cleaner.py -v

# With coverage report
pytest tests/ --cov=src/baligh --cov-report=html:htmlcov
```

---

### Test Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `config.py` | 100% | Critical; all paths must work |
| `data/cleaner.py` | 95% | Core pipeline; Arabic normalization critical |
| `data/mixer.py` | 90% | Ratio mixing must be exact |
| `data/formatter.py` | 95% | Tokenization correctness critical |
| `data/validators.py` | 90% | Schema validation must pass |
| `training/*` | 80% | Trainers tested via integration |
| `evaluation/` | 85% | Metrics must be accurate |
| `models/loader.py` | 85% | Model loading critical path |

**Overall Target**: 80% (enforced by `--cov-fail-under=80`)

---

### Test Patterns

#### Unit Test Pattern (AAA)
```python
def test_normalize_arabic_alef_variants():
    # Arrange
    from baligh.data.cleaner import normalize_arabic
    
    # Act
    result = normalize_arabic("أ إ آ ا")
    
    # Assert
    assert result == "ا ا ا ا"
```

#### Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.gpu
def test_cpt_trainer_smoke():
    # Uses max_samples=100 for fast test
    config = CPTConfig(max_steps=10, ...)
    trainer = CPTTrainer(config=config, train_dataset=tiny_dataset)
    result = trainer.train()
    assert result.training_loss < 10.0  # Sanity check
```

#### Fixtures
```python
# conftest.py
@pytest.fixture
def sample_cpt_data():
    return Dataset.from_dict({"text": ["نص تجريبي"] * 100})


@pytest.fixture
def sample_sft_data():
    return Dataset.from_dict(
        {"instruction": ["سؤال"] * 10, "input": [""] * 10, "output": ["جواب."] * 10}
    )
```

---

## Linting & Formatting

### Toolchain
| Tool | Version | Config |
|------|---------|--------|
| **Ruff** | 0.6.0 | `pyproject.toml` - `select = ["E","W","F","I","N","UP","B","C4","T20","SIM","ARG"]` |
| **Black** | 24.10.0 | `line-length=100`, `target-version=['py311','py312']` |
| **MyPy** | 1.11.0 | `strict=True`, `disallow_untyped_defs=True` |

### Commands
```bash
# Format
make format
# or
black src/ tests/
ruff check --fix src/

# Lint
make lint
# or
ruff check src/

# Type Check
make typecheck
# or
mypy src/
```

### Ruff Rules Explained
| Code | Rule | Why Enabled |
|------|------|-------------|
| E, W | pycodestyle | Basic style errors/warnings |
| F | pyflakes | Undefined names, unused imports |
| I | isort | Import sorting |
| N | pep8-naming | Naming conventions |
| UP | pyupgrade | Modern syntax upgrades |
| B | flake8-bugbear | Common bugs |
| C4 | flake8-comprehensions | Unnecessary comprehensions |
| T20 | flake8-print | No print() in production code |
| SIM | flake8-simplify | Simplifiable expressions |
| ARG | flake8-unused-arguments | Unused function arguments |

---

## Type Checking (MyPy)

### Configuration (`pyproject.toml`)
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
strict_optional = true
```

### Type Hints Standards
```python
# Good: Full type hints
def load_dataset_by_name(
    name: str, split: str | None = None, streaming: bool | None = None, **kwargs: Any
) -> Dataset: ...


# Good: Frozen dataclass with slots
@dataclass(frozen=True, slots=True)
class CPTConfig:
    max_steps: int = 50000
    learning_rate: float = 2e-4


# Avoid: Untyped defs (MyPy error)
def bad_example(x):  # ERROR: missing type hints
    return x * 2
```

---

## CI/CD Pipeline (GitHub Actions)

### Workflows (`.github/workflows/`)

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | Push/PR to main/develop | Lint → Typecheck → Unit Tests → Coverage |
| `cpt-training.yml` | Manual dispatch | CPT training (self-hosted GPU) |
| `sft-training.yml` | Manual dispatch | SFT training (self-hosted GPU) |
| `release.yml` | Manual dispatch (version tag) | Merge → Quantize → Push HF → GitHub Release |

### CI Pipeline (`ci.yml`)
```yaml
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install -r requirements/dev.txt && pip install -e .
      - run: ruff check src/
      - run: black --check src/
      - run: mypy src/
      - run: pytest tests/ -v --cov=src/baligh --cov-fail-under=80
```

### Pre-commit Hooks (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: https://github.com/psf/black
    hooks:
      - id: black
```

---

## Documentation Standards

### Docstring Style (NumPy/Google Hybrid)
```python
def load_base_model(
    model_name: str | None = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """Load base model with optional quantization.
    
    Args:
        model_name: Model name or path. Uses config default if None.
        load_in_4bit: Whether to load in 4-bit quantization.
        device_map: Device mapping strategy.
        
    Returns:
        Loaded model.
        
    Raises:
        ValueError: If model_name not found on HF Hub.
    """
```

### Module Docstrings
```python
"""Model loading utilities for Baligh-1.7B v0.

Handles 4-bit QLoRA loading, LoRA application/merging,
tokenizer loading, and model preparation for training.
"""
```

---

## Security Practices

### Secrets Management
- **`.env` in `.gitignore`** - Never committed
- **GitHub Secrets** - `HF_TOKEN`, `WANDB_API_KEY` for CI/CD
- **`SettingsConfigDict(extra="ignore")`** - Ignores unknown env vars

### Dependency Security
```bash
# Scan for vulnerabilities
pip-audit
safety check

# In CI
- run: pip-audit --desc --format=json
```

### Container Security
```dockerfile
# Non-root user
RUN addgroup -g 1001 -S appgroup && adduser -S appuser -u 1001
USER appuser

# Read-only filesystem where possible
# No secrets in image (injected at runtime)
```

---

## Code Review Checklist

### For Every PR
- [ ] All tests pass (`make test`)
- [ ] Linting clean (`make lint`)
- [ ] Type check passes (`make typecheck`)
- [ ] Coverage ≥ 80% (enforced by CI)
- [ ] No `print()` statements (use `logger`)
- [ ] No hardcoded paths (use `constants.py`)
- [ ] No magic numbers (use `constants.py` or config)
- [ ] Docstrings for public functions/classes
- [ ] Type hints on all public functions
- [ ] No `any` type without justification
- [ ] No `type: ignore` without comment
- [ ] Config changes reflected in YAML configs
- [ ] Breaking changes documented in CHANGELOG

### For ML-Specific PRs
- [ ] Config changes reflected in YAML configs
- [ ] New datasets added to `DATASET_REGISTRY` with metadata
- [ ] New metrics added to `evaluation/metrics.py`
- [ ] Benchmark changes documented in `docs/evaluation/`
- [ ] Model card template updated if metrics change

---

# Gotchas and Tips

## Critical Gotchas

### 1. **Padding Side Must Be Right for Causal LM**
```python
# ❌ WRONG - Breaks causal attention
tokenizer.padding_side = "left"

# ✅ CORRECT - Right padding for causal LM
tokenizer.padding_side = "right"
```
**Why**: Left padding shifts tokens; causal mask assumes future tokens are on the right

---

### 2. **Tokenizer Must Have pad_token**
```python
# Qwen has no pad_token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Standard workaround
```
**Forgetting this** → `ValueError: pad_token_id must be set` during training

---

### 3. **Response-only Loss Requires Accurate Masking**
```python
# ❌ WRONG - No masking; learns to predict user tokens
labels = input_ids.copy()

# ✅ CORRECT - Mask user tokens
labels = input_ids.copy()
assistant_start = find_assistant_start(input_ids)
labels[:assistant_start] = -100  # CrossEntropyLoss ignores -100
```
**Failure Mode**: Model learns to generate user prompts; fails at instruction following

---

### 4. **Streaming Datasets Cannot Use `num_proc > 1`**
```python
# ❌ WRONG - Streaming + multiprocessing = deadlock
dataset.map(fn, batched=True, num_proc=8)

# ✅ CORRECT
dataset.map(fn, batched=True, num_proc=1)  # Streaming
# OR
dataset.map(fn, batched=True, num_proc=8)  # Non-streaming (full load)
```

---

### 5. **`interleave_datasets` Stops at First Exhausted**
```python
# With stopping_strategy="first_exhausted" (default)
# Stops when SMALLEST dataset exhausted
# ArabicText-Large (743K) limits total steps vs ArabicWeb24 (28B tokens)
```
**Fix**: Ensure smallest dataset is large enough, or use `"all_exhausted"`

---

### 6. **LoRA Target Modules Must Match Model Architecture**
```python
# ❌ WRONG - Misses MLP projections
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ✅ CORRECT - Include MLP (SwiGLU) projections
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```
**Missing MLP** → LoRA only adapts attention; misses MLP knowledge

---

### 7. **`load_in_4bit` Requires `prepare_model_for_kbit_training`**
```python
# ❌ WRONG - Gradients won't flow properly
model = load_base_model(load_in_4bit=True)

# ✅ CORRECT
model = load_base_model(load_in_4bit=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
```
**Why**: Casts norms to fp32, enables gradient checkpointing for k-bit

---

### 8. **SFT `packing=False` is Mandatory**
```python
# ❌ WRONG - Packing mixes conversation turns
SFTConfig(packing=True, ...)

# ✅ CORRECT
SFTConfig(packing=False, ...)
```
**Why**: SFT conversations have semantic structure (user→assistant); packing concatenates unrelated turns

---

### 9. **SFT Loss: `response_only_loss=True` is Critical**
```python
# In SFTConfig
response_only_loss: bool = True  # Must be True!
```
**Without it**: Model learns to generate user prompts; fails instruction following

---

### 10. **`load_in_4bit=False` for Merging**
```python
# ❌ WRONG - Merging quantized model loses precision
model = load_base_model(load_in_4bit=True)
merged = merge_lora(model)

# ✅ CORRECT - Load full precision for merging
model = load_base_model(load_in_4bit=False)
merged = merge_lora(model)
```
**Why**: Merging requires full precision weights; 4-bit quantization loses precision irreversibly

---

## Performance Tips

### 1. **Use `HF_HUB_ENABLE_HF_TRANSFER=1` for Fast Uploads**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
# Multi-part parallel uploads; 5-10x faster for large models
```

### 2. **Use `uv` Instead of `pip`**
```bash
# 10-100x faster installs
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r requirements/training.txt
```

### 3. **Enable `torch.compile` (PyTorch 2.0+)**
```python
# In training script
model = torch.compile(model, mode="reduce-overhead")
# 10-20% speedup on Ampere+ GPUs
```

### 4. **Use `pad_to_multiple_of=8` for Tensor Cores**
```python
# In SFT data collator
DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
# Aligns to 8 for Tensor Core efficiency (8x8x8 MMA)
```

### 5. **Monitor GPU Memory with Callbacks**
```python
# In callbacks.py
class MemoryCallback:
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            log_memory_stats(prefix=f"Step {state.global_step}")
            clear_memory()
```

---

## Debugging Common Issues

### Issue: `CUDA Out of Memory`
**Solutions** (in order):
1. `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
2. `max_seq_length=1024` (reduce from 2048)
3. `gradient_checkpointing=True` (already default)
4. Disable `packing` for CPT (reduces peak memory)
5. Use `deepspeed` ZeRO-3 (if multi-GPU)

### Issue: `Triton Compilation Error`
```bash
# Unsloth requires Triton
pip install triton
# Or use --torch-backend=auto with uv
uv pip install unsloth --torch-backend=auto
```

### Issue: `CUDA Version Mismatch`
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall matching PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# or cu118 for older drivers
```

### Issue: `WandB Login Failed`
```bash
# Set API key
export WANDB_API_KEY=your_key
wandb login

# Or in script
wandb.login(key=os.getenv("WANDB_API_KEY"))
```

### Issue: `push_to_hub` Timeout
```bash
# Enable HF Transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# Retry with larger timeout
api.upload_folder(..., commit_message="...", timeout=3600)
```

---

## Pro Tips

### 1. **Test Pipeline with `max_samples=1000` First**
```bash
python -m src.scripts.prepare_data --stage cpt --clean --max-samples 1000
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --config configs/cpt/cpt-stage1.yaml
```
**Saves hours** catching config/data bugs early

### 2. **Use Config Files for Reproducibility**
```bash
# Instead of CLI args
python -m src.scripts.run_cpt --config configs/cpt/cpt-stage2.yaml
```
**Benefits**: Version-controlled config; exact reproducibility; easy grid search

### 3. **Monitor Memory Every 100 Steps**
```python
# In MemoryCallback
if state.global_step % 100 == 0:
    log_memory_stats(prefix=f"Step {state.global_step}")
    clear_memory()
```
**Catches OOM early** - shows allocation trend before crash

### 4. **Use `load_best_model_at_end=True` for SFT**
```yaml
# In SFT config
load_best_model_at_end: true
metric_for_best_model: "eval_loss"
greater_is_better: false
```
**Restores best checkpoint automatically** - no manual checkpoint selection

### 5. **Push to HF with `HF_TRANSFER`**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python -m src.scripts.push_to_hf --model-path release/Baligh-1.7B-v0-instruct --repo-id Kandil7/Baligh-1.7B
```
**10x faster uploads** for multi-GB models

### 6. **Validate Data Before Training**
```bash
# Quick sanity check
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/train_ready/cpt/train')
print(f'Examples: {len(ds)}')
print(f'Columns: {ds.column_names}')
print(f'Sample: {ds[0]}')
"
```
**Catches formatting bugs** before expensive training

### 7. **Use `make` Commands for Consistency**
```bash
make data-cpt      # Prepare CPT data
make train-cpt     # Run CPT training
make train-sft     # Run SFT training
make eval          # Run evaluation
make merge         # Merge LoRA
make quantize      # Quantize model
make push          # Push to HF
```

### 8. **Colab: Use T4 for Free, A100 for Speed**
| GPU | VRAM | CPT Time | SFT Time | Cost |
|-----|------|----------|----------|------|
| **T4 (Free)** | 16GB | ~4 hrs | ~2 hrs | Free |
| **L4 (Colab Pro)** | 24GB | ~2 hrs | ~1 hr | $10/mo |
| **A100 40GB** | 40GB | ~1 hr | ~30 min | $1-2/hr |

---

## Environment Setup Checklist

### Local Development
- [ ] Python 3.11+
- [ ] CUDA 12.1 + cuDNN 8.9+
- [ ] `make install-dev` (all deps)
- [ ] `.env` with `HF_TOKEN`, `WANDB_API_KEY`
- [ ] `pre-commit install`

### Colab
- [ ] Runtime → GPU (T4/L4)
- [ ] `HF_TOKEN` in Colab secrets
- [ ] `WANDB_API_KEY` in Colab secrets
- [ ] Mount Google Drive for persistence (optional)

### Production GPU (A100/H100)
- [ ] CUDA 12.1 + drivers 535+
- [ ] `make install-training`
- [ ] NVIDIA Container Toolkit (for Docker)
- [ ] HF_TOKEN in GitHub Secrets
- [ ] WANDB_API_KEY in GitHub Secrets

---

## Further Reading

| Topic | Resource |
|-------|----------|
| QLoRA Paper | https://arxiv.org/abs/2305.14314 |
| Unsloth Docs | https://unsloth.ai/docs/ |
| TRL SFTTrainer | https://huggingface.co/docs/trl/sft_trainer |
| Flash Attention 2 | https://github.com/Dao-AILab/flash-attention |
| Arabic NLP | https://github.com/arbml/arabic-nlp-resources |
| Islamic NLP | https://github.com/arbml/islamic-nlp |

---

# Summary

This documentation covers:

1. **Key Concepts Glossary** - Comprehensive terminology for Arabic NLP, Islamic knowledge, and ML training
2. **Engineering Decisions** - 13 detailed decisions with rationale, trade-offs, and consequences
3. **Code Quality Standards** - Testing, linting, CI/CD, documentation, security
4. **Gotchas and Tips** - 10 critical gotchas, 5 performance tips, 5 debugging solutions, 8 pro tips

The documentation teaches every concept from first principles and is specific to the Baligh-1.7B v0 project.
