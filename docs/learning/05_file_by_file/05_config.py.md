# config.py — Complete Line-by-Line Explanation

**File**: `src/baligh/config.py` (280 lines)
**Purpose**: Central configuration hub. Defines ALL configurable parameters for the entire Baligh project as typed, validated, immutable dataclasses.

---

## Imports (Lines 1-7)

```python
"""Configuration management for Baligh-1.7B v0."""
```
Line 1: Module docstring. Every Python module should have one.

```python
from pathlib import Path
```
Line 3: `Path` is Python's modern way to handle file system paths. Instead of string concatenation like `"data" + "/" + "raw"`, you write `Path("data") / "raw"`. The `/` operator is overloaded to join path segments. Using Path is cross-platform (works on Windows, Linux, Mac).

```python
from typing import Optional, Literal
```
Line 4: Type hints. `Optional[str]` means "a string or None". `Literal["cuda", "cpu", "mps"]` means "one of these exact strings" — Python will reject any other value at type-check time.

```python
from pydantic import Field, field_validator
```
Line 5: Pydantic tools. `Field` provides metadata for a field (default value, description, constraints). `field_validator` is a decorator that runs custom validation logic on a field before the object is created.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
```
Line 6: `BaseSettings` extends Pydantic's `BaseModel` with automatic loading from environment variables and `.env` files. `SettingsConfigDict` is the configuration for how settings are loaded.

```python
from dataclasses import dataclass, field
```
Line 7: Python's built-in `dataclass` decorator. Automatically generates `__init__`, `__repr__`, `__eq__` from class annotations. The `field` function is used for mutable default values (like dicts and lists) to avoid the "mutable default" bug.

---

## BaseConfig (Lines 10-55)

```python
class BaseConfig(BaseSettings):
```
Line 10: `BaseConfig` inherits from `BaseSettings` (Pydantic). This means:
- It can load values from environment variables (e.g., `SEED=42` in `.env`)
- It validates types automatically
- It has a `.env` file parser built in

```python
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
```
Lines 13-18: Configuration for the settings loader:
- `env_file=".env"`: Load variables from a `.env` file in the project root
- `env_file_encoding="utf-8"`: Read the file as UTF-8 (important for Arabic text)
- `case_sensitive=False`: `SEED`, `seed`, and `Seed` all map to the same field
- `extra="ignore"`: Ignore any environment variables that don't match a field (prevents errors from unrelated env vars)

### Project Fields (Lines 20-23)

```python
    project_name: str = "baligh"
```
Line 21: The project name. Used in logging, W&B runs, etc. Default is "baligh".

```python
    version: str = "0.1.0"
```
Line 22: Semantic version. Follows `MAJOR.MINOR.PATCH` format. v0.1.0 means "first development version".

```python
    environment: Literal["development", "staging", "production"] = "development"
```
Line 23: The deployment environment. `Literal` restricts to exactly these three strings. You cannot set `environment="test"` — Python will reject it.

### Path Fields (Lines 25-29)

```python
    data_dir: Path = Field(default=Path("data"))
```
Line 26: Root directory for all data. Default is `Path("data")` which resolves to `<project_root>/data/`. The `Field(default=...)` syntax is used instead of just `= Path("data")` because `Path` objects need special handling in Pydantic.

```python
    config_dir: Path = Field(default=Path("configs"))
```
Line 27: Where YAML config files live. Default: `configs/`.

```python
    output_dir: Path = Field(default=Path("training"))
```
Line 28: Where training outputs (checkpoints, logs) go. Default: `training/`.

```python
    logs_dir: Path = Field(default=Path("training/logs"))
```
Line 29: Where log files are written. Default: `training/logs/`.

### Hardware Fields (Lines 31-34)

```python
    device: Literal["cuda", "cpu", "mps"] = "cuda"
```
Line 32: Which compute device to use. `cuda` = NVIDIA GPU, `cpu` = processor, `mps` = Apple Silicon GPU. Default is CUDA.

```python
    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"
```
Line 33: Precision for training. `bf16` (bfloat16) is the default because it has a larger dynamic range than `fp16`, making training more stable. `no` means full float32 (slowest, most memory).

```python
    gradient_checkpointing: bool = True
```
Line 34: Enable gradient checkpointing. This trades ~30% more compute for ~70% less activation memory. Essential for training on consumer GPUs.

### Distributed Training Fields (Lines 36-39)

```python
    world_size: int = 1
```
Line 37: Number of GPUs for distributed training. Default is 1 (single GPU).

```python
    local_rank: int = 0
```
Line 38: Which GPU this process uses. In single-GPU mode, always 0.

```python
    ddp_backend: str = "nccl"
```
Line 39: Backend for distributed training. `nccl` is optimized for NVIDIA GPUs. `gloo` is used for CPU training.

### Logging Fields (Lines 41-46)

```python
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
```
Line 42: Logging verbosity. `INFO` shows normal operations. `DEBUG` shows everything. `WARNING` shows only problems.

```python
    log_format: Literal["json", "text"] = "text"
```
Line 43: Whether logs are human-readable text or machine-parseable JSON.

```python
    wandb_project: Optional[str] = "Baligh-1.7B"
```
Line 44: Weights & Biases project name. `Optional[str]` means it can be a string or None. If None, W&B logging is disabled.

```python
    wandb_entity: Optional[str] = None
```
Line 45: W&B team/entity name. None means use your personal account.

```python
    wandb_api_key: Optional[str] = None
```
Line 46: W&B API key. Should be set via environment variable, not in code.

### Reproducibility Fields (Lines 48-50)

```python
    seed: int = 42
```
Line 49: Random seed. 42 is the "Answer to the Ultimate Question of Life, the Universe, and Everything" (Hitchhiker's Guide). Same seed = same results every run.

```python
    deterministic: bool = True
```
Line 50: Whether to use deterministic CUDA operations. True gives reproducible results but may be slower.

### Path Resolution Validator (Lines 52-55)

```python
    @field_validator("data_dir", "config_dir", "output_dir", "logs_dir", mode="before")
```
Line 52: Decorator that runs validation on these four fields. `mode="before"` means it runs before Pydantic's default validation. This converts string paths to absolute paths.

```python
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()
```
Lines 53-55: The validator function. Takes a string or Path, converts to Path, then calls `.resolve()` which makes it absolute (e.g., `"data"` becomes `"/home/user/Baligh/data"`).

---

## ModelConfig (Lines 58-87)

```python
@dataclass(frozen=True, slots=True)
class ModelConfig:
```
Lines 58-59: `frozen=True` means the object is immutable after creation (you cannot change `model_name` after creating a ModelConfig). `slots=True` saves memory by not creating a `__dict__` for each instance.

```python
    model_name: str = "Qwen/Qwen3-1.7B"
```
Line 63: The Hugging Face model ID. This is what gets passed to `AutoModelForCausalLM.from_pretrained()`. Qwen3-1.7B is a 1.54B parameter model by Alibaba.

```python
    unsloth_model_name: str = "unsloth/Qwen3-1.7B-Base"
```
Line 64: The Unsloth-optimized version. This model is already pre-quantized to 4-bit with BitsAndBytes, making it faster to load for QLoRA training.

```python
    trust_remote_code: bool = True
```
Line 65: Allow loading model code from the Hugging Face repo. Required for Qwen2.5 because it uses custom modeling code.

```python
    tokenizer_name: Optional[str] = None  # None = same as model
```
Line 68: Which tokenizer to use. None means "use the same tokenizer as the model". Since Qwen2.5's tokenizer is bundled with the model, this is correct.

```python
    max_seq_length: int = 2048
```
Line 69: Maximum sequence length for training. 2048 tokens. The model supports 32K but training at 2048 to save memory. Can be extended later.

```python
    padding_side: Literal["left", "right"] = "right"
```
Line 70: Where to add padding tokens. Right padding is standard for decoder-only models (padding at the end, not the beginning).

```python
    truncation_side: Literal["left", "right"] = "right"
```
Line 71: Where to truncate long sequences. Right truncation removes tokens from the end, preserving the beginning (which contains the prompt/instruction).

```python
    load_in_4bit: bool = True
```
Line 74: Load model weights in 4-bit precision. This is the "Q" in QLoRA — quantized loading.

```python
    load_in_8bit: bool = False
```
Line 75: Alternative 8-bit loading. Disabled because 4-bit is more memory-efficient.

```python
    bnb_4bit_compute_dtype: str = "bfloat16"
```
Line 76: Data type for computation during 4-bit training. bfloat16 is used because it has the same range as float32 but half the memory.

```python
    bnb_4bit_quant_type: str = "nf4"
```
Line 77: Quantization type. NF4 (Normal Float 4) is optimized for normally-distributed neural network weights. Better than FP4.

```python
    bnb_4bit_use_double_quant: bool = True
```
Line 78: Double quantization. Quantizes the quantization constants themselves. Saves ~0.4 GB per billion parameters with negligible quality loss.

```python
    use_cache: bool = False  # Disable for training
```
Line 81: KV cache for faster inference. Disabled during training because it wastes memory (we don't need cached key-value pairs during training).

```python
    attn_implementation: Optional[str] = "flash_attention_2"  # or "sdpa"
```
Line 82: Attention implementation. Flash Attention 2 is ~2x faster and uses ~50% less memory than standard attention. Falls back to SDPA on older GPUs.

```python
    def __post_init__(self):
        if self.tokenizer_name is None:
            object.__setattr__(self, "tokenizer_name", self.model_name)
```
Lines 84-86: Called after the dataclass is initialized. If no tokenizer name was specified, use the model name. Uses `object.__setattr__` because the dataclass is frozen (normal `self.tokenizer_name = ...` would fail).

---

## LoRAConfig (Lines 89-105)

```python
@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """LoRA/QLoRA configuration."""
```
Lines 89-91: LoRA = Low-Rank Adaptation. Instead of fine-tuning all 1.54B parameters, we only train ~0.1% of them through small adapter matrices.

```python
    r: int = 16
```
Line 93: Rank of the LoRA matrices. Higher = more expressive but more parameters. 16 is a good balance.

```python
    lora_alpha: int = 16
```
Line 94: Scaling factor. The effective scaling is `alpha/r`. With alpha=16, r=16, scaling=1.0 (no scaling).

```python
    lora_dropout: float = 0.0
```
Line 95: Dropout for LoRA layers. 0.0 means no dropout. Unsloth optimizes better without dropout.

```python
    bias: Literal["none", "all", "lora_only"] = "none"
```
Line 96: Whether to train bias terms. "none" means only the LoRA matrices are trained, not any bias terms.

```python
    task_type: str = "CAUSAL_LM"
```
Line 97: Task type for PEFT. "CAUSAL_LM" = causal language modeling (predict next token).

```python
target_modules: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
```
Lines 98-101: Which layers get LoRA adapters. These are:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`: Attention projection layers (Query, Key, Value, Output)
- `gate_proj`, `up_proj`, `down_proj`: MLP (Feed-Forward) layers

This targets ALL major linear layers, giving maximum expressiveness.

```python
    modules_to_save: Optional[tuple[str, ...]] = None
```
Line 102: Additional modules to save (not LoRA adapters). None means only LoRA adapters are saved.

```python
    init_lora_weights: bool = True
```
Line 103: Initialize LoRA weights properly (Kaiming uniform for A, zeros for B).

```python
    use_rslora: bool = False
```
Line 104: Rank-Stabilized LoRA. An alternative initialization that can help with higher ranks. Not needed at r=16.

```python
    use_dora: bool = False
```
Line 105: DoRA (Weight-Decomposed Low-Rank Adaptation). A newer method that can improve quality but isn't needed here.

---

## CPTConfig (Lines 108-155)

```python
@dataclass(frozen=True, slots=True)
class CPTConfig:
    """Continued Pretraining configuration."""
```
Lines 108-110: CPT = Continued Pretraining. This is Stage 1: adapting the base model to Arabic domain by training on raw Arabic text.

```python
dataset_mix: dict[str, float] = field(
    default_factory=lambda: {
        "arabicweb24": 0.70,
        "arabictext_large": 0.20,
        "arabic_pile": 0.10,
    }
)
```
Lines 113-117: Dataset mixing ratios. `default_factory=lambda: {...}` is used because dicts are mutable — you cannot use `= {...}` directly as a default value in a dataclass.
- arabicweb24: 70% (largest, web-crawled Arabic text)
- arabictext_large: 20% (general Arabic text)
- arabic_pile: 10% (mixed Arabic dialects)

```python
    islamic_cycle: bool = True
```
Line 118: Whether to run a separate training cycle on Islamic datasets after the main CPT.

```python
islamic_datasets: tuple[str, ...] = (
    "hadith_datasets",
    "quran_qa",
    "quran_md",
    "arabic_islamic_texts",
)
```
Lines 119-121: Which Islamic datasets to use in the cycle.

### Training Hyperparameters (Lines 123-132)

```python
    max_steps: int = 50000
```
Line 124: Maximum training steps. 50K steps at batch size 8 = ~400M tokens per epoch.

```python
    num_train_epochs: Optional[float] = None
```
Line 125: Number of epochs. None means "train for max_steps only, ignore epochs".

```python
    per_device_train_batch_size: int = 2
```
Line 126: Batch size per GPU. With gradient_accumulation_steps=4, effective batch = 2*4 = 8.

```python
    gradient_accumulation_steps: int = 4
```
Line 127: Accumulate gradients over 4 steps before updating weights. This simulates a larger batch size without using more memory.

```python
    learning_rate: float = 2e-4
```
Line 128: Learning rate. 2e-4 = 0.0002. This is relatively high for fine-tuning but appropriate for CPT because we're adapting a lot of knowledge.

```python
    weight_decay: float = 0.01
```
Line 129: L2 regularization. Prevents weights from growing too large. 0.01 is standard.

```python
    warmup_steps: int = 1000
```
Line 130: Linearly increase learning rate from 0 to 2e-4 over the first 1000 steps. Prevents early training instability.

```python
    lr_scheduler_type: Literal["linear", "cosine", "constant", "cosine_with_restarts"] = "cosine"
```
Line 131: Learning rate schedule. Cosine smoothly decreases LR following a cosine curve. Best for stable training.

```python
    max_grad_norm: float = 1.0
```
Line 132: Gradient clipping. If the gradient norm exceeds 1.0, scale it down. Prevents exploding gradients.

### Optimization (Lines 134-138)

```python
    optim: str = "adamw_torch_fused"
```
Line 135: Optimizer. `adamw_torch_fused` is the fused AdamW implementation in PyTorch. 15-20% faster than standard AdamW on CUDA.

```python
    adam_beta1: float = 0.9
```
Line 136: Exponential moving average of gradients. 0.9 means "mostly look at recent gradients".

```python
    adam_beta2: float = 0.95
```
Line 137: Exponential moving average of squared gradients. 0.95 is slightly higher than the default 0.999, giving more stable updates.

```python
    adam_epsilon: float = 1e-8
```
Line 138: Small constant to prevent division by zero.

### Logging and Saving (Lines 140-145)

```python
    logging_steps: int = 10
```
Line 141: Log metrics every 10 steps.

```python
    save_steps: int = 500
```
Line 142: Save checkpoint every 500 steps.

```python
    save_total_limit: int = 3
```
Line 143: Keep only the last 3 checkpoints. Older ones are deleted to save disk space.

```python
    eval_steps: int = 500
```
Line 144: Evaluate on validation set every 500 steps.

```python
    evaluation_strategy: Literal["no", "steps", "epoch"] = "steps"
```
Line 145: When to evaluate. "steps" = every eval_steps. "epoch" = end of each epoch. "no" = never.

### Data Loading (Lines 147-151)

```python
    dataloader_num_workers: int = 4
```
Line 148: Number of parallel data loading processes. 4 workers load data while the GPU trains.

```python
    dataloader_pin_memory: bool = True
```
Line 149: Pin memory for faster CPU-to-GPU transfer. Uses page-locked memory.

```python
    dataloader_drop_last: bool = True
```
Line 150: Drop the last incomplete batch. Prevents small batches at the end of training.

```python
    preprocessing_num_workers: int = 8
```
Line 151: Number of workers for data preprocessing (tokenization, cleaning).

### Packing (Lines 153-155)

```python
    packing: bool = True
```
Line 154: Pack multiple short sequences into one sample. Eliminates padding waste. 2-5x throughput improvement for CPT.

```python
    packing_max_length: Optional[int] = None  # None = max_seq_length
```
Line 155: Maximum length for packed sequences. None means use max_seq_length (2048).

---

## SFTConfig (Lines 158-205)

```python
@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Supervised Fine-Tuning configuration."""
```
Lines 158-160: SFT = Supervised Fine-Tuning. Stage 2: teaching the model to follow instructions.

```python
dataset_mix: dict[str, float] = field(
    default_factory=lambda: {
        "cidar": 0.40,
        "evol_instruct_arabic": 0.35,
        "gazelle": 0.10,
        "summarization": 0.10,
        "islamic_qa": 0.05,
    }
)
```
Lines 163-169: SFT dataset ratios. CIDAR (40%) is the largest Arabic instruction dataset. evol_instruct_arabic (35%) provides diverse instructions. Gazelle (10%) adds Arabic writing tasks. Summarization (10%) adds summarization ability. Islamic QA (5%) adds Islamic knowledge.

### Key Differences from CPTConfig

```python
    max_steps: int = 10000
```
Line 172: Only 10K steps for SFT (vs 50K for CPT). SFT needs less data because it's teaching format, not knowledge.

```python
    learning_rate: float = 1e-4  # Lower than CPT
```
Line 176: Lower learning rate (1e-4 vs 2e-4). SFT requires more careful updates to avoid catastrophic forgetting.

```python
    warmup_steps: int = 500
```
Line 178: Shorter warmup (500 vs 1000). Less data means shorter warmup is appropriate.

```python
    packing: bool = False  # Usually False for SFT to maintain conversation structure
```
Line 189: Packing is OFF for SFT. Packing would merge separate conversations, corrupting the chat template and response-only loss.

```python
    response_only_loss: bool = True  # Only compute loss on assistant responses
```
Line 190: Critical for SFT. Loss is only computed on assistant tokens, not user tokens. The model learns to generate responses, not predict user messages.

```python
    load_best_model_at_end: bool = True
```
Line 198: After training, load the checkpoint with the best eval_loss.

```python
    metric_for_best_model: str = "eval_loss"
```
Line 199: Which metric determines "best". Lower eval_loss = better.

```python
    greater_is_better: bool = False
```
Line 200: False means lower metric is better (loss should decrease).

---

## EvalConfig (Lines 208-239)

```python
@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Evaluation configuration."""
```
Lines 208-210: Configuration for running benchmarks.

```python
eval_datasets: tuple[str, ...] = (
    "mmlu_arabic",
    "cidar_eval",
    "cidar_mcq",
    "mr_tydi_arabic",
    "islamic_qa_custom",
)
```
Lines 213-216: Which benchmarks to run. MMLU-Arabic for knowledge, CIDAR for instruction-following, mr-tydi for retrieval QA, Islamic QA for domain knowledge.

### Generation Parameters (Lines 218-225)

```python
    max_new_tokens: int = 512
```
Line 219: Maximum tokens to generate per response.

```python
    temperature: float = 0.7
```
Line 220: Sampling temperature. 0.7 = somewhat creative but not random. 1.0 = full randomness. 0.0 = greedy (deterministic).

```python
    top_p: float = 0.9
```
Line 221: Nucleus sampling. Only consider tokens in the top 90% probability mass.

```python
    top_k: int = 50
```
Line 222: Only consider the top 50 most likely tokens.

```python
    repetition_penalty: float = 1.1
```
Line 223: Penalize repeated tokens. 1.1 = 10% penalty for each repetition.

```python
    do_sample: bool = True
```
Line 224: Use sampling (True) or greedy decoding (False). True gives more diverse outputs.

```python
    num_beams: int = 1
```
Line 225: Beam search. 1 = no beam search (just sample). Higher values explore more possibilities but are slower.

### Human Evaluation (Lines 231-239)

```python
human_eval_rubric: dict[str, str] = field(
    default_factory=lambda: {
        "correctness": "1-5: Factual accuracy of the response",
        "clarity": "1-5: Clarity and readability of Arabic",
        "arabic_quality": "1-5: Formal Arabic quality (fusha)",
        "usefulness": "1-5: Helpfulness for the user's query",
        "faithfulness": "1-5: Faithfulness to source/knowledge",
    }
)
```
Lines 233-239: Human evaluation rubric. Each response is scored 1-5 on five dimensions. This provides qualitative evaluation beyond automated metrics.

---

## QuantizationConfig (Lines 242-262)

```python
@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Quantization configuration for model export."""
```
Lines 242-244: Configuration for post-training quantization (reducing model size for deployment).

```python
    gguf_quantization: str = "q4_k_m"  # q4_k_m, q5_k_m, q8_0, etc.
```
Line 247: GGUF format quantization. q4_k_m = 4-bit quantization with k-quant method, medium quality. Best balance of size and quality.

```python
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    awq_version: str = "gemm"
```
Lines 250-253: AWQ (Activation-aware Weight Quantization) settings. 4-bit, group size 128, zero point enabled, GEMM kernel for fast inference.

```python
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = True
```
Lines 256-258: GPTQ settings. 4-bit, group size 128, desc_act = quantize activations in descending order of importance.

```python
    calibration_samples: int = 512
    calibration_seq_length: int = 2048
```
Lines 261-262: Calibration data for quantization. 512 samples of 2048 tokens each.

---

## Factory Functions (Lines 265-280)

```python
def get_config() -> BaseConfig:
    """Get the base configuration singleton."""
    return BaseConfig()
```
Lines 265-267: Creates a new BaseConfig each time (loads from .env). Despite the docstring saying "singleton", it's NOT a singleton — each call creates a fresh instance.

```python
def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return ModelConfig()
```
Lines 270-272: Creates a new ModelConfig with defaults.

```python
def get_lora_config() -> LoRAConfig:
    """Get LoRA configuration."""
    return LoRAConfig()
```
Lines 275-277: Creates a new LoRAConfig with defaults.

```python
def get_cpt_config() -> CPTConfig:
    """Get CPT configuration."""
    return CPTConfig()
```
Lines 280+: Creates a new CPTConfig with defaults.

---

## Design Decisions

1. **Frozen dataclasses**: Prevents accidental mutation during training (configs are passed around extensively).
2. **Factory functions over singletons**: Each call creates fresh instances, avoiding state-sharing bugs.
3. **Pydantic BaseSettings for BaseConfig**: Needs env var parsing and validation.
4. **Literal types**: Restricts values to known-good options, catching typos at type-check time.
5. **field(default_factory=...) for mutable defaults**: Prevents the classic Python "mutable default" bug.
