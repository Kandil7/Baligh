"""Configuration management for Baligh-1.5B v0."""

from pathlib import Path
from typing import Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass, field


class BaseConfig(BaseSettings):
    """Base configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Project
    project_name: str = "baligh"
    version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    
    # Paths
    data_dir: Path = Field(default=Path("data"))
    config_dir: Path = Field(default=Path("configs"))
    output_dir: Path = Field(default=Path("training"))
    logs_dir: Path = Field(default=Path("training/logs"))
    
    # Hardware
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"
    gradient_checkpointing: bool = True
    
    # Distributed
    world_size: int = 1
    local_rank: int = 0
    ddp_backend: str = "nccl"
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"
    wandb_project: Optional[str] = "baligh-1.5b"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    @field_validator("data_dir", "config_dir", "output_dir", "logs_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model architecture and loading configuration."""
    
    # Base model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    unsloth_model_name: str = "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit"
    trust_remote_code: bool = True
    
    # Tokenizer
    tokenizer_name: Optional[str] = None  # None = same as model
    max_seq_length: int = 2048
    padding_side: Literal["left", "right"] = "right"
    truncation_side: Literal["left", "right"] = "right"
    
    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Model config
    use_cache: bool = False  # Disable for training
    attn_implementation: Optional[str] = "flash_attention_2"  # or "sdpa"
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            object.__setattr__(self, "tokenizer_name", self.model_name)


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """LoRA/QLoRA configuration."""
    
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )
    modules_to_save: Optional[tuple[str, ...]] = None
    init_lora_weights: bool = True
    use_rslora: bool = False
    use_dora: bool = False


@dataclass(frozen=True, slots=True)
class CPTConfig:
    """Continued Pretraining configuration."""
    
    # Data
    dataset_mix: dict[str, float] = field(default_factory=lambda: {
        "arabicweb24": 0.70,
        "arabictext_large": 0.20,
        "arabic_pile": 0.10,
    })
    islamic_cycle: bool = True
    islamic_datasets: tuple[str, ...] = (
        "hadith_datasets", "quran_qa", "quran_md", "arabic_islamic_texts"
    )
    
    # Training
    max_steps: int = 50000
    num_train_epochs: Optional[float] = None
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_scheduler_type: Literal["linear", "cosine", "constant", "cosine_with_restarts"] = "cosine"
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    evaluation_strategy: Literal["no", "steps", "epoch"] = "steps"
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    preprocessing_num_workers: int = 8
    
    # Packing
    packing: bool = True
    packing_max_length: Optional[int] = None  # None = max_seq_length


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Supervised Fine-Tuning configuration."""
    
    # Data
    dataset_mix: dict[str, float] = field(default_factory=lambda: {
        "cidar": 0.40,
        "evol_instruct_arabic": 0.35,
        "gazelle": 0.10,
        "summarization": 0.10,
        "islamic_qa": 0.05,
    })
    
    # Training
    max_steps: int = 10000
    num_train_epochs: Optional[float] = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4  # Lower than CPT
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = "cosine"
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # SFT specific
    packing: bool = False  # Usually False for SFT to maintain conversation structure
    response_only_loss: bool = True  # Only compute loss on assistant responses
    
    # Logging & saving
    logging_steps: int = 5
    save_steps: int = 250
    save_total_limit: int = 3
    eval_steps: int = 250
    evaluation_strategy: Literal["no", "steps", "epoch"] = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    preprocessing_num_workers: int = 8


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Evaluation configuration."""
    
    # Datasets
    eval_datasets: tuple[str, ...] = (
        "mmlu_arabic", "cidar_eval", "cidar_mcq", 
        "mr_tydi_arabic", "islamic_qa_custom"
    )
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    
    # Perplexity
    perplexity_batch_size: int = 8
    perplexity_stride: int = 512
    
    # Human eval
    human_eval_samples: int = 100
    human_eval_rubric: dict[str, str] = field(default_factory=lambda: {
        "correctness": "1-5: Factual accuracy of the response",
        "clarity": "1-5: Clarity and readability of Arabic",
        "arabic_quality": "1-5: Formal Arabic quality (fusha)",
        "usefulness": "1-5: Helpfulness for the user's query",
        "faithfulness": "1-5: Faithfulness to source/knowledge",
    })


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Quantization configuration for model export."""
    
    # GGUF
    gguf_quantization: str = "q4_k_m"  # q4_k_m, q5_k_m, q8_0, etc.
    
    # AWQ
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    awq_version: str = "gemm"
    
    # GPTQ
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = True
    
    # General
    calibration_samples: int = 512
    calibration_seq_length: int = 2048


def get_config() -> BaseConfig:
    """Get the base configuration singleton."""
    return BaseConfig()


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return ModelConfig()


def get_lora_config() -> LoRAConfig:
    """Get LoRA configuration."""
    return LoRAConfig()


def get_cpt_config() -> CPTCon
