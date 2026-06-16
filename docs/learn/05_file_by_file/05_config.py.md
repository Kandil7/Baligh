# config.py - Configuration Management

**Path**: `src/baligh/config.py` (280 lines)

## Purpose

Central configuration hub for the entire Baligh project. Defines all configurable parameters as typed, validated, immutable dataclasses with environment variable support.

---

## Classes

### BaseConfig (lines 10-55)

Pydantic `BaseSettings` subclass. Loads from `.env` files and environment variables.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| project_name | str | "baligh" | Project identifier |
| version | str | "0.1.0" | Semantic version |
| environment | Literal | "development" | Deployment environment |
| data_dir | Path | "data" | Data storage root |
| config_dir | Path | "configs" | YAML config root |
| output_dir | Path | "training" | Training outputs root |
| logs_dir | Path | "training/logs" | Log files root |
| device | Literal | "cuda" | Compute device |
| mixed_precision | Literal | "bf16" | Precision mode |
| gradient_checkpointing | bool | True | Memory optimization |
| seed | int | 42 | Random seed |
| deterministic | bool | True | Deterministic CUDA |
| wandb_project | Optional[str] | "baligh-1.5b" | W&B project name |
| log_level | Literal | "INFO" | Logging verbosity |

The `resolve_path` field validator (line 52) converts string paths to absolute `Path` objects at construction time.

---

### ModelConfig (lines 58-87)

Frozen dataclass for model architecture settings.

| Field | Default | Rationale |
|-------|---------|-----------|
| model_name | "Qwen/Qwen2.5-1.5B" | HF model ID |
| unsloth_model_name | "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit" | Unsloth 4-bit variant |
| max_seq_length | 2048 | Training context (extendable to 32K) |
| load_in_4bit | True | QLoRA quantization |
| bnb_4bit_quant_type | "nf4" | Normal Float 4 (best quality) |
| bnb_4bit_use_double_quant | True | Double quantization for extra compression |
| attn_implementation | "flash_attention_2" | Memory-efficient attention |
| use_cache | False | Disabled during training |

The `__post_init__` (line 84) sets `tokenizer_name` to `model_name` when not explicitly provided.

---

### LoRAConfig (lines 89-105)

Frozen dataclass for LoRA adapter settings.

| Field | Default | Rationale |
|-------|---------|-----------|
| r | 16 | Rank - balance expressiveness vs params |
| lora_alpha | 16 | Scaling (alpha/r = 1.0 means no scaling) |
| lora_dropout | 0.0 | No dropout (Unsloth optimization) |
| bias | "none" | No bias terms in adapters |
| target_modules | 7 modules | All attention + MLP projections |
| init_lora_weights | True | Proper initialization |

Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

---

### CPTConfig (lines 108-156)

Frozen dataclass for Continued Pretraining.

Key fields: `dataset_mix` (dict of name to ratio), `max_steps` (50000), `learning_rate` (2e-4), `packing` (True), `warmup_steps` (1000), `lr_scheduler_type` ("cosine").

---

### SFTConfig (lines 158-206)

Frozen dataclass for Supervised Fine-Tuning.

Key differences from CPTConfig: `learning_rate` is lower (1e-4), `packing` is False (preserve conversation structure), `response_only_loss` is True, `load_best_model_at_end` is True.

---

### EvalConfig (lines 208-239)

Frozen dataclass for evaluation settings.

Includes benchmark list, generation parameters (temperature, top_p, top_k, repetition_penalty), perplexity settings, and human eval rubric definitions.

---

### QuantizationConfig (lines 242-262)

Frozen dataclass for export quantization (GGUF, AWQ, GPTQ).

---

## Factory Functions (lines 265-280)

- `get_config()` -> BaseConfig
- `get_model_config()` -> ModelConfig
- `get_lora_config()` -> LoRAConfig
- `get_cpt_config()` -> CPTConfig
- `get_sft_config()` -> SFTConfig
- `get_eval_config()` -> EvalConfig
- `get_quantization_config()` -> QuantizationConfig

Each returns a fresh instance with defaults. No singletons.
