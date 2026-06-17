"""Model loading utilities for Baligh-1.5B v0."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from baligh.config import get_lora_config, get_model_config
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats

logger = get_logger(__name__)


def load_base_model(
    model_name: str | None = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = "flash_attention_2",
    use_cache: bool = False,
) -> PreTrainedModel:
    """Load base model with optional quantization.

    Args:
        model_name: Model name or path. Uses config default if None.
        load_in_4bit: Whether to load in 4-bit quantization.
        load_in_8bit: Whether to load in 8-bit quantization.
        device_map: Device mapping strategy.
        torch_dtype: Torch dtype for model weights.
        attn_implementation: Attention implementation.
        use_cache: Whether to use KV cache.

    Returns:
        Loaded model.
    """
    config = get_model_config()
    model_name = model_name or config.unsloth_model_name

    logger.info(f"Loading model: {model_name}")
    log_memory_stats(prefix="Before model load")

    # Prepare quantization config
    from transformers import BitsAndBytesConfig

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    # Load model
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype if not load_in_4bit else torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=config.trust_remote_code,
        use_cache=use_cache,
    )

    # Prepare for k-bit training
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    log_memory_stats(prefix="After model load")
    assert model.config is not None
    logger.info(f"Model loaded: {model.config.model_type}")

    return model


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig | None = None,
) -> PeftModel:
    """Apply LoRA/QLoRA to model.

    Args:
        model: Base model.
        lora_config: LoRA configuration. Uses default if None.

    Returns:
        Model with LoRA adapters.
    """
    if lora_config is None:
        cfg = get_lora_config()
        lora_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            task_type=cfg.task_type,
            target_modules=list(cfg.target_modules),
            modules_to_save=list(cfg.modules_to_save) if cfg.modules_to_save else None,
            init_lora_weights=cfg.init_lora_weights,
            use_rslora=cfg.use_rslora,
            use_dora=cfg.use_dora,
        )

    logger.info(f"Applying LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_lora_model(
    base_model: PreTrainedModel,
    lora_path: str,
    is_trainable: bool = False,
) -> PeftModel:
    """Load LoRA adapters onto base model.

    Args:
        base_model: Base model.
        lora_path: Path to LoRA adapters.
        is_trainable: Whether adapters should be trainable.

    Returns:
        Model with loaded LoRA adapters.
    """
    logger.info(f"Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=is_trainable)
    return model


def merge_lora(
    model: PeftModel,
    save_path: str | None = None,
) -> PreTrainedModel:
    """Merge LoRA adapters into base model.

    Args:
        model: PEFT model with LoRA adapters.
        save_path: Optional path to save merged model.

    Returns:
        Merged base model.
    """
    logger.info("Merging LoRA adapters...")
    merged_model = model.merge_and_unload()

    if save_path:
        logger.info(f"Saving merged model to: {save_path}")
        merged_model.save_pretrained(save_path, safe_serialization=True)

    return merged_model


def load_tokenizer(
    tokenizer_name: str | None = None,
    max_seq_length: int = 2048,
    padding_side: str = "right",
    truncation_side: str = "right",
) -> PreTrainedTokenizerBase:
    """Load tokenizer.

    Args:
        tokenizer_name: Tokenizer name or path. Uses model config if None.
        max_seq_length: Maximum sequence length.
        padding_side: Padding side.
        truncation_side: Truncation side.

    Returns:
        Loaded tokenizer.
    """
    config = get_model_config()
    tokenizer_name = tokenizer_name or config.tokenizer_name

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )

    # Configure tokenizer
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Set model max length
    tokenizer.model_max_length = max_seq_length

    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    return tokenizer


def prepare_model_for_training(
    model: PreTrainedModel,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """Prepare model for training.

    Args:
        model: Model to prepare.
        gradient_checkpointing: Enable gradient checkpointing.

    Returns:
        Prepared model.
    """
    # Enable gradient checkpointing
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        logger.info("Gradient checkpointing enabled")

    # Disable cache for training
    if model.config is not None:
        model.config.use_cache = False

    return model


def get_model_info(model: PreTrainedModel) -> dict:
    """Get model information.

    Args:
        model: Model to inspect.

    Returns:
        Dictionary with model info.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = model.config
    assert config is not None
    return {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
