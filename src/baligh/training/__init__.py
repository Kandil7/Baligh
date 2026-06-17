"""Training package for Baligh-1.5B v0."""

from baligh.training.callbacks import CheckpointCallback, LoggingCallback, MemoryCallback
from baligh.training.checkpoint import (
    CheckpointManager,
    get_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from baligh.training.lora_config import create_lora_config, create_quantization_config
from baligh.training.metrics import compute_metrics, compute_perplexity

__all__ = [
    "LoggingCallback",
    "MemoryCallback",
    "CheckpointCallback",
    "CheckpointManager",
    "compute_metrics",
    "compute_perplexity",
    "create_lora_config",
    "create_quantization_config",
    "save_checkpoint",
    "load_checkpoint",
    "get_latest_checkpoint",
]


def __getattr__(name):
    if name in ("CPTTrainer", "train_cpt"):
        from baligh.training.cpt_trainer import CPTTrainer, train_cpt
        return {"CPTTrainer": CPTTrainer, "train_cpt": train_cpt}[name]
    if name in ("SFTTrainer", "train_sft"):
        from baligh.training.sft_trainer import SFTTrainer, train_sft
        return {"SFTTrainer": SFTTrainer, "train_sft": train_sft}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
