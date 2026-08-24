# 07 — Engineering Decisions: Baligh-1.7B v0

## Decision 1: Qwen3-1.7B as Base Model

**Chosen**: Qwen3-1.7B Base (non-instruct)

**Alternatives considered**:
- Arabic-only models (OASIS, AraGPT2): Too small, limited evaluation, no GQA
- Multilingual models (BLOOM, BERT-Arabic): Decoder-only preferred for generation
- Larger Qwen variants (3B, 7B): Too expensive for consumer GPU training

**Rationale**: Qwen2.5 has the best open Arabic support at the 1.5B scale. GQA (Grouped Query Attention) provides efficient inference. 32K max context (trained at 2048, extendable later). The base (non-instruct) variant is correct for CPT since we want to adapt the base model, not fine-tune an already-instructed model.

**Risk**: Qwen2.5 is by Alibaba (Chinese company). Community trust and long-term support are unknown. Mitigated by the model being open-weight on HF.

## Decision 2: Two-Stage Training (CPT then SFT)

**Chosen**: CPT on raw Arabic text, then SFT on instruction pairs

**Alternatives considered**:
- SFT only (skip CPT): Less Arabic domain adaptation
- CPT only (skip SFT): No instruction-following capability
- Single-stage combined: Harder to tune, less control

**Rationale**: CPT first teaches the model to "think in Arabic" by exposing it to billions of Arabic tokens. SFT then teaches it to follow instructions. This two-stage approach is standard practice and provides better results than either stage alone.

## Decision 3: QLoRA 4-bit Over Full Fine-Tuning

**Chosen**: QLoRA with NF4 quantization, rank=16

**Alternatives considered**:
- Full fine-tuning: Requires 12+ GB VRAM, impractical on consumer GPUs
- LoRA without quantization: ~8 GB VRAM, still tight
- QLoRA 8-bit: Less compression, more VRAM than needed
- Higher rank (r=32, 64): More parameters, diminishing returns

**Rationale**: QLoRA 4-bit reduces VRAM to ~4 GB, enabling training on a single RTX 4090. Rank=16 provides good expressiveness with minimal parameters (~0.1% of total). NF4 is superior to FP4 for neural network weights. Double quantization saves another ~0.4 GB.

**Trade-off**: QLoRA may produce slightly lower quality than full fine-tuning. For a 1.5B model, this difference is negligible compared to the accessibility gains.

## Decision 4: Frozen Dataclasses for Configs

**Chosen**: `@dataclass(frozen=True, slots=True)` for all config classes

**Alternatives considered**:
- Pydantic BaseModel: More validation features, but heavier
- Plain dicts: No type safety, easy to mutate accidentally
- Named tuples: Immutable but no defaults or validation
- Regular dataclasses: Mutable, risk of accidental changes

**Rationale**: Frozen dataclasses provide immutability (configs shouldn't change during training), memory efficiency (slots=True), and hashability (can be dict keys). The BaseConfig uses Pydantic because it needs env var parsing, but all training/eval configs use frozen dataclasses.

## Decision 5: TRL SFTTrainer Over HF Trainer for SFT

**Chosen**: TRL's SFTTrainer for the SFT stage

**Alternatives considered**:
- HF Trainer with manual masking: More code, error-prone
- Custom training loop: Too much boilerplate
- Axolotl/LlamaFactory: Additional dependency, less control

**Rationale**: TRL's SFTTrainer natively supports response-only loss (masks user tokens), packing, and chat templates. It integrates cleanly with the existing PEFT/transformers stack. The `formatting_func` parameter allows our PromptFormatter to be plugged in directly.

## Decision 6: Packing in CPT, Not in SFT

**Chosen**: Packing=True for CPT, Packing=False for SFT

**Rationale**: CPT uses raw text where multiple short documents can be safely concatenated. Packing eliminates padding waste and improves throughput 2-5x. SFT uses conversation-formatted data where packing would merge separate conversations, corrupting the chat structure and response-only loss computation.

## Decision 7: Streaming for Large Datasets

**Chosen**: Streaming mode for ArabicWeb24 (28B tokens) and Arabic Pile (5B tokens)

**Alternatives considered**:
- Download full datasets: 30+ GB disk space, hours of download
- Use smaller subsets: Loses data diversity

**Rationale**: Streaming loads data lazily from HF Hub, eliminating the need to download 30+ GB. Trade-off is that streaming doesn't support random access (needed for shuffling), but `interleave_datasets` with `first_exhausted` strategy provides sufficient randomization.

## Decision 8: Loguru Over Stdlib Logging

**Chosen**: Loguru for all logging

**Alternatives considered**:
- Python stdlib logging: Verbose setup, no rotation built-in
- structlog: Good but more complex API
- Custom logger: Reinventing the wheel

**Rationale**: Loguru provides clean API, built-in rotation (100MB, 30 days), JSON serialization, color output, and automatic stdlib interception. One-line setup vs 20+ lines with stdlib.

## Decision 9: Response-Only Loss for SFT

**Chosen**: Loss computed only on assistant tokens

**Alternatives considered**:
- Standard causal LM loss on all tokens: Teaches model to predict user inputs too, wastes compute
- Custom masking: Error-prone, harder to maintain

**Rationale**: Response-only loss focuses training signal on what matters: producing good assistant responses. TRL's SFTTrainer handles this automatically when using chat templates. The model should learn to generate responses, not predict user messages.

## Decision 10: Islamic QA at 5% of SFT Mix

**Chosen**: 5% Islamic QA in SFT dataset mix

**Rationale**: Islamic knowledge is a differentiator but should not dominate instruction-following capability. 5% provides enough signal for Islamic queries without overshadowing general Arabic instruction-following (cidar=40%, evol_instruct=35%). The CPT Islamic cycle (separate training pass on hadith/quran datasets) provides the bulk of Islamic knowledge at the pretraining level.

## Decision 11: Cosine LR Schedule for Both Stages

**Chosen**: Cosine with warmup for both CPT and SFT

**Alternatives considered**:
- Linear decay: Simpler but less smooth
- Constant LR: No decay, may cause late-training instability
- Cosine with restarts: More complex, not needed for this scale

**Rationale**: Cosine provides smooth decay that prevents the learning rate from dropping too quickly (which can underfit) or too slowly (which can cause instability). Warmup steps (1000 for CPT, 500 for SFT) prevent early training instability when gradients are noisy.

## Decision 12: eval_loss as Best Model Metric

**Chosen**: `metric_for_best_model="eval_loss"` with `greater_is_better=False`

**Alternatives considered**:
- Perplexity: Same signal as loss (perplexity = exp(loss)), but HF expects loss
- Custom metrics: More complex, harder to compute during training
- No early stopping: Risk of overfitting

**Rationale**: eval_loss is the simplest and most reliable metric during training. It directly measures the model's prediction quality on held-out data. Lower is better, so `greater_is_better=False`. The `load_best_model_at_end=True` setting ensures we keep the checkpoint with the lowest eval loss.
