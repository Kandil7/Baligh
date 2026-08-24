"""Memory management utilities for training."""

import gc

import torch

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def log_memory_stats(device: int | torch.device | None = None, prefix: str = "") -> dict:
    """Log current GPU memory statistics.

    Args:
        device: GPU device index or torch.device
        prefix: Prefix for log message

    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()

    stats = torch.cuda.memory_stats(device)

    allocated = stats.get("allocated_bytes.all.current", 0) / 1e9
    reserved = stats.get("reserved_bytes.all.current", 0) / 1e9
    max_allocated = stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_reserved = stats.get("reserved_bytes.all.peak", 0) / 1e9

    logger.info(
        f"{prefix} GPU Memory: "
        f"Allocated={allocated:.2f}GB, "
        f"Reserved={reserved:.2f}GB, "
        f"Max Allocated={max_allocated:.2f}GB, "
        f"Max Reserved={max_reserved:.2f}GB"
    )

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "max_reserved_gb": max_reserved,
    }


def clear_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_model_memory(model: torch.nn.Module) -> None:
    """Print model parameter memory usage."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = (param_size + buffer_size) / 1e9

    logger.info(
        f"Model Memory: "
        f"Params={param_size / 1e9:.2f}GB, "
        f"Buffers={buffer_size / 1e9:.2f}GB, "
        f"Total={total_size:.2f}GB"
    )


def estimate_training_memory(
    model_params: int,
    batch_size: int,
    seq_length: int,
    precision: str = "fp16",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
    trainable_fraction: float | None = None,
) -> dict:
    """Estimate memory requirements for training.

    Args:
        model_params: Number of model parameters.
        batch_size: Per-device batch size.
        seq_length: Sequence length.
        precision: Model precision (fp16, bf16, fp32, int8, int4).
        optimizer: Optimizer type (adamw, adamw_8bit, adam, sgd).
        gradient_checkpointing: Whether gradient checkpointing is enabled.
        trainable_fraction: Fraction of parameters receiving gradients and
            optimizer states (e.g. ~0.01 for LoRA r=16 on a 1.5B model).
            If None, assumed to be 1.0 (full fine-tuning).

    Returns:
        Dictionary with memory estimates in GB.
    """
    # Model weights
    bytes_per_param = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int4": 0.5}
    model_mem = model_params * bytes_per_param.get(precision, 2) / 1e9

    fraction = trainable_fraction if trainable_fraction is not None else 1.0
    fraction = max(0.0, min(1.0, fraction))

    # Gradients only exist for trainable params (LoRA adapters in QLoRA).
    grad_mem = model_mem * fraction

    # Optimizer states apply to trainable params only. AdamW keeps fp32
    # moments (~2x fp16 param size); scale relative to the *trainable* share.
    opt_multiplier = {"adamw": 8, "adamw_8bit": 4, "sgd": 4, "adam": 8}
    opt_mem = (model_params * fraction * bytes_per_param.get("fp32", 4) / 1e9) * (
        opt_multiplier.get(optimizer, 8) / 8
    )

    # Activations
    # Rough estimate: batch * seq * hidden * layers * bytes
    # For Qwen3-1.7B: hidden=2048, layers=28
    hidden_size = 2048
    num_layers = 28
    activation_bytes = bytes_per_param.get(precision, 2)
    activation_mem = (
        batch_size * seq_length * hidden_size * num_layers * activation_bytes * 4 / 1e9
    )  # factor for intermediate activations

    if gradient_checkpointing:
        activation_mem *= 0.3  # Roughly 70% reduction

    # Total
    total = model_mem + grad_mem + opt_mem + activation_mem

    return {
        "model_gb": round(model_mem, 3),
        "gradients_gb": round(grad_mem, 3),
        "optimizer_gb": round(opt_mem, 3),
        "activations_gb": round(activation_mem, 3),
        "total_gb": round(total, 3),
        "assumes": ("full fine-tuning" if fraction == 1.0 else f"trainable_fraction={fraction}"),
    }


class MemoryTracker:
    """Track memory usage during training."""

    def __init__(self, device: int | torch.device | None = None) -> None:
        resolved: int | torch.device | None
        if device is None:
            resolved = torch.cuda.current_device() if torch.cuda.is_available() else None
        else:
            resolved = device
        self.device: int | torch.device | None = resolved
        self.snapshots: list[dict] = []

    def snapshot(self, label: str) -> dict:
        """Take a memory snapshot (no-op dict on CPU-only systems)."""
        stats = log_memory_stats(self.device, prefix=f"[{label}]")
        if not stats:
            # CPU-only: nothing measurable; keep an empty placeholder so
            # summary() can distinguish "no data" from real zero usage.
            return {}
        stats["label"] = label
        self.snapshots.append(stats)
        return stats

    def summary(self) -> dict:
        """Get memory usage summary."""
        usable = [s for s in self.snapshots if s]
        if not usable:
            return {}

        allocated = [s["allocated_gb"] for s in usable]
        reserved = [s["reserved_gb"] for s in usable]

        return {
            "peak_allocated_gb": max(allocated),
            "peak_reserved_gb": max(reserved),
            "avg_allocated_gb": sum(allocated) / len(allocated),
            "avg_reserved_gb": sum(reserved) / len(reserved),
            "snapshots": self.snapshots,
        }
