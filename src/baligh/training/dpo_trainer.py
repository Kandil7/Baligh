"""Direct Preference Optimization Trainer for Baligh-1.7B v0.

Wraps TRL's ``DPOTrainer`` for the DPO/ORPO loss computation.

Why TRL: the DPO loss needs frozen-reference logprobs for chosen and
rejected on every batch; re-implementing that numerics is error-prone.
TRL is the maintained reference implementation.

Design:
- ``base_model_path`` (the SFT output) is applied BEFORE model loading —
  same load-bearing contract as SFTTrainer.
- With a PEFT/LoRA policy, TRL derives reference logits by disabling
  adapters (no second copy of the weights), so ``ref_model=None`` is the
  memory-efficient default on 16 GB.
- All trainer kwargs go through TRL's ``DPOConfig`` (a TrainingArguments
  subclass). Passing beta/loss_type as bare constructor kwargs was removed
  in trl>=0.9 and raises TypeError there.
"""

from pathlib import Path
from typing import Any

from baligh.config import get_config, get_dpo_config, get_model_config
from baligh.models.loader import apply_lora, load_base_model
from baligh.models.tokenizer import get_tokenizer
from baligh.training.checkpoint import CheckpointManager
from baligh.training.cpt_trainer import resolve_reporters
from baligh.utils.hardware import resolve_precision
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


class DPOTrainer:
    """DPO/ORPO trainer backed by TRL's DPOTrainer."""

    def __init__(
        self,
        config: Any = None,
        model: Any = None,
        ref_model: Any = None,
        tokenizer: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        output_dir: str | Path | None = None,
        base_model_path: str | Path | None = None,
    ) -> None:
        self.config = config or get_dpo_config()
        self.model_config = get_model_config()
        self.base_config = get_config()

        # Continuation source must be set BEFORE _setup_model loads weights.
        if base_model_path:
            logger.info(f"DPO will continue from SFT checkpoint: {base_model_path}")
            object.__setattr__(self.model_config, "model_name", str(base_model_path))

        self.output_dir = Path(output_dir) if output_dir else (self.base_config.output_dir / "dpo")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.base_config.seed)

        self.tokenizer = tokenizer or get_tokenizer(
            self.model_config.tokenizer_name, self.model_config.max_seq_length
        )

        if model is None:
            self.model = self._setup_model()
        else:
            self.model = model

        # None + PEFT policy => TRL uses adapter-disabled forward as reference.
        self.ref_model = ref_model

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.checkpoint_manager = CheckpointManager(
            self.output_dir, keep_last_n=self.config.save_total_limit
        )
        self.trainer = self._create_trainer()
        self.checkpoint_manager.install_signal_handler(self.trainer)

        logger.info("DPOTrainer initialized")
        log_memory_stats(prefix="After trainer init")

    @classmethod
    def from_config(cls, config_dict: dict, **kwargs: Any) -> "DPOTrainer":
        """Create trainer from config dictionary."""
        from baligh.config import DPOConfig

        config = DPOConfig(**config_dict.get("dpo", {}))
        return cls(config=config, **kwargs)

    def _setup_model(self) -> Any:
        """Load policy checkpoint + LoRA."""
        return apply_lora(
            load_base_model(
                model_name=self.model_config.model_name,
                load_in_4bit=self.model_config.load_in_4bit,
                load_in_8bit=self.model_config.load_in_8bit,
                use_cache=False,
                is_inference=False,
            )
        )

    def _training_kwargs(self) -> dict[str, Any]:
        precision = resolve_precision(self.base_config.mixed_precision)
        max_steps = self.config.max_steps
        configured_epochs = self.config.num_train_epochs
        num_train_epochs: float | None = (
            None
            if (max_steps is not None and max_steps > 0)
            else (configured_epochs if configured_epochs is not None else 1.0)
        )
        return {
            "output_dir": str(self.output_dir),
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "max_grad_norm": self.config.max_grad_norm,
            "optim": self.config.optim,
            "adam_beta1": self.config.adam_beta1,
            "adam_beta2": self.config.adam_beta2,
            "adam_epsilon": self.config.adam_epsilon,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "save_total_limit": self.config.save_total_limit,
            "eval_steps": self.config.eval_steps,
            "eval_strategy": self.config.eval_strategy,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "dataloader_pin_memory": self.config.dataloader_pin_memory,
            "remove_unused_columns": False,
            "report_to": resolve_reporters(
                self.base_config.wandb_project, self.base_config.wandb_api_key
            ),
            "run_name": "baligh-1.7b-dpo",
            "seed": self.base_config.seed,
            "data_seed": self.base_config.seed,
            "bf16": precision == "bf16",
            "fp16": precision == "fp16",
            "gradient_checkpointing": True,
            "ddp_find_unused_parameters": False,
        }

    def _create_trainer(self) -> Any:
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer as TRLDPOTrainer

        kwargs = self._training_kwargs()
        kwargs.update(
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            label_smoothing=self.config.label_smoothing,
            reference_free=self.config.reference_free,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        dpo_args = TRLDPOConfig(**kwargs)

        return TRLDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            peft_config=None,  # LoRA already applied to the policy
        )

    def train(self, resume_from_checkpoint: str | None = None) -> Any:
        logger.info("Starting DPO training")
        log_memory_stats(prefix="Before training")

        try:
            result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        finally:
            self.checkpoint_manager.uninstall_signal_handler()

        log_memory_stats(prefix="After training")
        logger.info(f"Training completed: {result}")

        self.save_model()
        return result

    def save_model(self, path: str | Path | None = None) -> None:
        save_path = Path(path) if path else self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_path}")
        self.trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]
        logger.info("Model saved")


def train_dpo(
    train_dataset: Any,
    eval_dataset: Any = None,
    output_dir: str | Path | None = None,
    resume_from_checkpoint: str | None = None,
    base_model_path: str | Path | None = None,
    config: dict | None = None,
) -> Any:
    """Run DPO alignment. ``base_model_path`` points at the SFT output to
    continue from — it is load-bearing, not decorative."""
    trainer_kwargs: dict[str, Any] = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "output_dir": output_dir,
        "base_model_path": base_model_path,
    }
    trainer = (
        DPOTrainer.from_config(config, **trainer_kwargs) if config else DPOTrainer(**trainer_kwargs)
    )
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
