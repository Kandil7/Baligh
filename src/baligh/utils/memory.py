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
    precision: str = "bf16",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> dict:
    """Estimate memory requirements for training.

    Args:
        model_params: Number of model parameters
        batch_size: Per-device batch size
        seq_length: Sequence length
        precision: Model precision (fp16, bf16, fp32)
        optimizer: Optimizer type
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        Dictionary with memory estimates in GB
    """
    # Model weights
    bytes_per_param = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int4": 0.5}
    model_mem = model_params * bytes_per_param.get(precision, 2) / 1e9

    # Gradients (same size as model)
    grad_mem = model_mem

    # Optimizer states (AdamW: 2x model size for fp32 states)
    opt_multiplier = {"adamw": 2, "adamw_8bit": 1, "sgd": 1, "adam": 2}
    opt_mem = model_mem * opt_multiplier.get(optimizer, 2)

    # Activations
    # Rough estimate: batch * seq * hidden * layers * bytes
    # For Qwen2.5-1.5B: hidden=2048, layers=28
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
        "model_gb": model_mem,
        "gradients_gb": grad_mem,
        "optimizer_gb": opt_mem,
        "activations_gb": activation_mem,
        "total_gb": total,
    }


class MemoryTracker:
    """Track memory usage during training."""

    def __init__(self, device: int | torch.device | None = None):
        self.device = device or torch.cuda.current_device()
        self.snapshots = []

    def snapshot(self, label: str) -> dict:
        """Take a memory snapshot."""
        stats = log_memory_stats(self.device, prefix=f"[{label}]")
        stats["label"] = label
        self.snapshots.append(stats)
        return stats

    def summary(self) -> dict:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}

        allocated = [s["allocated_gb"] for s in self.snapshots]
        reserved = [s["reserved_gb"] for s in self.snapshots]

        return {
            "peak_allocated_gb": max(allocated),
            "peak_reserved_gb": max(reserved),
            "avg_allocated_gb": sum(allocated) / len(allocated),
            "avg_reserved_gb": sum(reserved) / len(reserved),
            "snapshots": self.snapshots,
        }
