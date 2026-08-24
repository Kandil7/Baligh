"""Model loading utilities for Baligh-1.7B v0.

Precision and attention defaults are resolved against the actual GPU at
load time: bf16/FlashAttention-2 only on sm_80+ (Ampere+), fp16/sdpa on
Turing (Colab T4, Quadro RTX 5000). See baligh.utils.hardware.

Two load modes:
- Training (default): 4-bit NF4 + k-bit preparation (gradient checkpointing,
  input hooks), KV cache off.
- Inference (``is_inference=True``): quantized weights but NO training prep,
  KV cache on — generation without a KV cache recomputes the prefix per token.
"""

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from baligh.config import get_lora_config, get_model_config
from baligh.utils.hardware import (
    resolve_attn_implementation,
    resolve_precision,
    resolve_torch_dtype,
)
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats

logger = get_logger(__name__)


def build_quantization_config(
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytes config from ModelConfig defaults."""
    if not (load_in_4bit or load_in_8bit):
        return None
    model_config = get_model_config()
    if compute_dtype is None:
        compute_dtype = resolve_torch_dtype(resolve_precision("auto"))
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_base_model(
    model_name: str | None = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    torch_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
    use_cache: bool = False,
    is_inference: bool = False,
) -> PreTrainedModel:
    """Load the base model with hardware-appropriate settings.

    Args:
        model_name: Model name or path. Uses config ``model_name``
            (unsloth/Qwen3-1.7B-Base) if None.
        load_in_4bit: Load in 4-bit NF4.
        load_in_8bit: Load in 8-bit (ignored when 4-bit requested).
        device_map: Device mapping strategy.
        torch_dtype: Weight dtype. Auto-resolved per GPU capability if None.
        attn_implementation: Explicit attention backend. Auto-resolved
            (flash_attention_2 on sm_80+, sdpa otherwise) if None.
        use_cache: KV-cache flag for non-inference loads.
        is_inference: Skip all training preparation and force KV cache on.

    Returns:
        Loaded model, prepared for its intended mode.
    """
    config = get_model_config()
    model_name = model_name or config.model_name

    if torch_dtype is None:
        torch_dtype = resolve_torch_dtype(resolve_precision("auto"))
    if attn_implementation is None:
        attn_implementation = resolve_attn_implementation(config.attn_implementation)

    effective_use_cache = True if is_inference else use_cache
    logger.info(
        f"Loading model: {model_name} "
        f"(4bit={load_in_4bit}, dtype={torch_dtype}, attn={attn_implementation}, "
        f"inference={is_inference})"
    )
    log_memory_stats(prefix="Before model load")

    quantization_config = build_quantization_config(load_in_4bit, load_in_8bit, torch_dtype)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=config.trust_remote_code,
        use_cache=effective_use_cache,
    )

    # K-bit preparation is a TRAINING-only step: it upcasts norms, hooks
    # inputs for gradient flow, and enables gradient checkpointing. Running
    # it on inference models disables the KV cache and wastes memory.
    if not is_inference and (load_in_4bit or load_in_8bit):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    log_memory_stats(prefix="After model load")
    assert model.config is not None
    logger.info(f"Model loaded: {model.config.model_type}")

    return model


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig | None = None,
) -> PeftModel:
    """Attach LoRA adapters (built from baligh.training.lora_config by default)."""
    if lora_config is None:
        from baligh.training.lora_config import create_lora_config

        lora_config = create_lora_config()

    logger.info(f"Applying LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    model = get_peft_model(model, lora_config)  # type: ignore[assignment]
    model.print_trainable_parameters()  # type: ignore[union-attr]
    return model  # type: ignore[return-value]


def load_lora_model(
    base_model: PreTrainedModel,
    lora_path: str,
    is_trainable: bool = False,
) -> PeftModel:
    """Load trained LoRA adapters onto a base model."""
    logger.info(f"Loading LoRA from: {lora_path}")
    return PeftModel.from_pretrained(base_model, lora_path, is_trainable=is_trainable)


def load_model_with_adapter(
    model_path_or_name: str | None = None,
    adapter_path: str | None = None,
    is_inference: bool = True,
) -> PreTrainedModel | PeftModel:
    """Convenience: load the configured base, then attach adapters if given."""
    base = load_base_model(model_name=model_path_or_name, is_inference=is_inference)
    if adapter_path:
        return load_lora_model(base, adapter_path, is_trainable=False)
    return base


def merge_lora(
    model: PeftModel,
    save_path: str | None = None,
) -> PreTrainedModel:
    """Merge LoRA adapters into the base weights.

    The model MUST be a PEFT-wrapped model loaded in fp16/bf16 — adapters
    cannot be merged into bnb-4bit linear layers. Use
    ``load_base_model(..., load_in_4bit=False)`` +
    ``load_lora_model(...)`` to construct the input.
    """
    if not isinstance(model, PeftModel):
        raise TypeError(
            "merge_lora expects a PeftModel (call load_lora_model() first). "
            f"Got {type(model).__name__} — there are no adapters to merge."
        )
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
    """Delegates to :func:`baligh.models.tokenizer.get_tokenizer`.

    Kept as an alias so existing imports keep working; behavior lives in
    exactly one place.
    """
    from baligh.models.tokenizer import get_tokenizer

    return get_tokenizer(tokenizer_name, max_seq_length, padding_side, truncation_side)


def prepare_model_for_training(
    model: PreTrainedModel,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """Enable gradient checkpointing and disable cache for training."""
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        logger.info("Gradient checkpointing enabled")
    if model.config is not None:
        model.config.use_cache = False
    return model


def get_model_info(model: PreTrainedModel) -> dict:
    """Get parameter counts and architecture facts for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = model.config
    assert config is not None
    return {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
