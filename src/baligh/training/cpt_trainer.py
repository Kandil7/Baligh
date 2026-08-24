"""Continued Pretraining Trainer for Baligh-1.7B v0.

Expects PRE-FORMATTED datasets (columns: input_ids / attention_mask /
labels) produced by ``baligh.data.prepare_data`` or
``PromptFormatter.format_cpt_batch``.

Stack: transformers ``Trainer`` + PEFT QLoRA. Unsloth is NOT used on this
code path (it remains available in the Colab notebooks).
"""

from typing import Any

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from baligh.config import get_config, get_cpt_config, get_model_config
from baligh.models.loader import apply_lora, load_base_model
from baligh.models.tokenizer import get_tokenizer
from baligh.training.callbacks import LoggingCallback, MemoryCallback
from baligh.training.checkpoint import CheckpointManager
from baligh.utils.hardware import resolve_precision
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


def resolve_reporters(
    wandb_project: str | None = None, wandb_api_key: str | None = None
) -> list[str]:
    """Pick reporting backends that cannot block training.

    wandb without an API key (and not explicitly offline) hangs on an
    interactive login prompt — exactly what you don't want mid-run on a
    rented GPU. Default to tensorboard; opt into wandb deliberately.
    """
    import os

    if wandb_project and (wandb_api_key or os.environ.get("WANDB_MODE") == "offline"):
        return ["wandb", "tensorboard"]
    if wandb_project:
        logger.warning(
            "wandb_project is set but no API key found (set WANDB_API_KEY or "
            "WANDB_MODE=offline) — reporting to tensorboard only"
        )
    return ["tensorboard"]


class CPTTrainer:
    def __init__(
        self,
        config: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        output_dir: Any = None,
    ) -> None:
        self.config = config or get_cpt_config()
        self.model_config = get_model_config()
        self.base_config = get_config()
        self.output_dir = output_dir or self.base_config.output_dir / "cpt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.base_config.seed)

        self.tokenizer = tokenizer or get_tokenizer(
            self.model_config.tokenizer_name, self.model_config.max_seq_length
        )

        if model is None:
            self.model = self._setup_model()
        else:
            self.model = model

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.checkpoint_manager = CheckpointManager(
            self.output_dir, keep_last_n=self.config.save_total_limit
        )

        self.training_args = self._create_training_args()
        self.trainer = self._create_trainer()

        self.checkpoint_manager.install_signal_handler(self.trainer)

        logger.info("CPTTrainer initialized")
        log_memory_stats(prefix="After trainer init")

    @classmethod
    def from_config(cls, config_dict: dict, **kwargs: Any) -> "CPTTrainer":
        """Create trainer from config dictionary."""
        from baligh.config import CPTConfig

        config = CPTConfig(**config_dict.get("cpt", {}))
        return cls(config=config, **kwargs)

    def _setup_model(self) -> Any:
        """Load base + LoRA. K-bit preparation happens once, inside
        load_base_model — repeating it post-LoRA double-upcasts norms."""
        return apply_lora(
            load_base_model(
                model_name=self.model_config.model_name,
                load_in_4bit=self.model_config.load_in_4bit,
                load_in_8bit=self.model_config.load_in_8bit,
                use_cache=False,
                is_inference=False,
            )
        )

    def _create_training_args(self) -> TrainingArguments:
        precision = resolve_precision(self.base_config.mixed_precision)
        # max_steps > 0 takes precedence over epochs in HF Trainer; passing
        # both is misleading. Only forward the one that will actually drive
        # the schedule.
        max_steps = self.config.max_steps
        configured_epochs = self.config.num_train_epochs
        num_train_epochs: float | None = (
            None
            if (max_steps is not None and max_steps > 0)
            else (configured_epochs if configured_epochs is not None else 1.0)
        )
        return TrainingArguments(
            output_dir=str(self.output_dir),
            max_steps=max_steps,
            # HF accepts None (epochs mode); the stub over-restricts.
            num_train_epochs=num_train_epochs,  # type: ignore[arg-type]
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_steps=self.config.eval_steps,
            eval_strategy=self.config.eval_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_drop_last=self.config.dataloader_drop_last,
            remove_unused_columns=False,
            report_to=resolve_reporters(
                self.base_config.wandb_project, self.base_config.wandb_api_key
            ),
            run_name="baligh-1.7b-cpt",
            seed=self.base_config.seed,
            data_seed=self.base_config.seed,
            bf16=precision == "bf16",
            fp16=precision == "fp16",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
        )

    def _create_trainer(self) -> Trainer:
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer: Trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=[LoggingCallback(), MemoryCallback(log_every_n_steps=100)],
        )
        return trainer

    def train(self, resume_from_checkpoint: str | None = None) -> Any:
        logger.info("Starting CPT training")
        log_memory_stats(prefix="Before training")

        try:
            result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        finally:
            self.checkpoint_manager.uninstall_signal_handler()

        log_memory_stats(prefix="After training")
        logger.info(f"Training completed: {result}")

        self.save_model()
        return result

    def save_model(self, path: Any = None) -> None:
        save_path = path or self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_path}")
        self.trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]
        logger.info("Model saved")


def train_cpt(
    train_dataset: Any,
    eval_dataset: Any = None,
    output_dir: Any = None,
    resume_from_checkpoint: str | None = None,
    config: dict | None = None,
) -> Any:
    if config:
        trainer = CPTTrainer.from_config(
            config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
        )
    else:
        trainer = CPTTrainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
        )
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
