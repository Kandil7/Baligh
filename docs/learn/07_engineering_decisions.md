# Engineering Decisions - Baligh-1.5B v0

## Decision Template

Each decision follows: **Decision → Context → Options → Chosen → Trade-offs → Consequences**

---

## 1. Base Model: Qwen2.5-1.5B Base (not Instruct)

**Decision**: Use Qwen2.5-1.5B **Base** (pretrained) not Instruct

**Context**: Need Arabic-first model with Islamic specialization; starting point determines capabilities

**Options Considered**:
| Option | Pros | Cons |
|--------|------|------|
| Qwen2.5-1.5B Base | Clean slate; full control over Arabic adaptation | Requires full CPT + SFT |
| Qwen2.5-1.5B Instruct | Ready for chat; less training | Baked-in English alignment; harder to Arabic-ize |
| LLaMA-2-1.5B | - | Doesn't exist |
| Mistral-1.5B | - | Doesn't exist |
| BLOOM-1.5B | Multilingual | Weak Arabic; deprecated |

**Chosen**: **Qwen2.5-1.5B Base**

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
- Pre-quantized models (`unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit`)
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
