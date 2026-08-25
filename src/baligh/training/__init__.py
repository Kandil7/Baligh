"""Training package for Baligh-1.7B v0."""

from typing import Any

from baligh.training.callbacks import CheckpointCallback, LoggingCallback, MemoryCallback
from baligh.training.checkpoint import CheckpointManager
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
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy exports
    if name in ("CPTTrainer", "train_cpt"):
        from baligh.training.cpt_trainer import CPTTrainer, train_cpt

        return {"CPTTrainer": CPTTrainer, "train_cpt": train_cpt}[name]
    if name in ("SFTTrainer", "train_sft"):
        from baligh.training.sft_trainer import SFTTrainer, train_sft

        return {"SFTTrainer": SFTTrainer, "train_sft": train_sft}[name]
    if name in ("DPOTrainer", "train_dpo"):
        from baligh.training.dpo_trainer import DPOTrainer, train_dpo

        return {"DPOTrainer": DPOTrainer, "train_dpo": train_dpo}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
