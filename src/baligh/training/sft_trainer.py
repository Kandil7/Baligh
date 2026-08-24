"""Supervised Fine-Tuning Trainer for Baligh-1.7B v0.

Design (v0.2 rewrite):
- Plain ``transformers.Trainer`` over a PRE-FORMATTED dataset
  (input_ids / attention_mask / labels) built by
  ``baligh.data.get_sft_formatter().map(batched=True)``. TRL's SFTTrainer
  was dropped: its API churn (tokenizer=, formatting_func=,
  dataset_text_field conflicts) broke this repo three separate ways, and
  pre-formatting gives us explicit completion-only loss masking.
- ``base_model_path`` is honored: SFT continues from the CPT output.
  Passing it used to be silently ignored, which disconnected the CPT->SFT
  pipeline entirely.
"""

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from baligh.config import get_config, get_model_config, get_sft_config
from baligh.models.loader import apply_lora, load_base_model
from baligh.models.tokenizer import get_tokenizer
from baligh.training.callbacks import LoggingCallback, MemoryCallback
from baligh.training.checkpoint import CheckpointManager
from baligh.training.cpt_trainer import resolve_reporters
from baligh.utils.hardware import resolve_precision
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)


class SFTTrainer:
    def __init__(
        self,
        config=None,
        model=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        output_dir=None,
        base_model_path=None,
    ):
        self.config = config or get_sft_config()
        self.model_config = get_model_config()
        self.base_config = get_config()
        # Explicit continuation source (e.g. training/cpt/final) wins over
        # the configured base; log loudly so stage handoffs are auditable.
        if base_model_path:
            logger.info(f"SFT will continue from: {base_model_path}")
            object.__setattr__(self.model_config, "model_name", base_model_path)
        self.output_dir = output_dir or self.base_config.output_dir / "sft"
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

        logger.info("SFTTrainer initialized")
        log_memory_stats(prefix="After trainer init")

    @classmethod
    def from_config(cls, config_dict: dict, **kwargs):
        """Create trainer from config dictionary."""
        from baligh.config import SFTConfig as BalighSFTConfig

        config = BalighSFTConfig(**config_dict.get("sft", {}))
        return cls(config=config, **kwargs)

    def _setup_model(self):
        """Load base + LoRA. K-bit preparation happens once inside
        load_base_model; repeating it post-LoRA double-upcasts norms."""
        return apply_lora(
            load_base_model(
                model_name=self.model_config.model_name,
                load_in_4bit=self.model_config.load_in_4bit,
                load_in_8bit=self.model_config.load_in_8bit,
                use_cache=False,
                is_inference=False,
            )
        )

    def _create_training_args(self):
        precision = resolve_precision(self.base_config.mixed_precision)
        max_steps = self.config.max_steps
        num_train_epochs = (
            None
            if (max_steps is not None and max_steps > 0)
            else (self.config.num_train_epochs or 1.0)
        )
        return TrainingArguments(
            output_dir=str(self.output_dir),
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
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
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=False,
            report_to=resolve_reporters(
                self.base_config.wandb_project, self.base_config.wandb_api_key
            ),
            run_name="baligh-1.7b-sft",
            seed=self.base_config.seed,
            data_seed=self.base_config.seed,
            bf16=precision == "bf16",
            fp16=precision == "fp16",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
        )

    def _create_trainer(self):
        # Pads input_ids with pad_token and labels with -100 so masked
        # prompt positions never contribute to loss.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        )
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=[LoggingCallback(), MemoryCallback(log_every_n_steps=100)],
        )

    def train(self, resume_from_checkpoint=None):
        logger.info("Starting SFT training")
        log_memory_stats(prefix="Before training")

        try:
            result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        finally:
            self.checkpoint_manager.uninstall_signal_handler()

        log_memory_stats(prefix="After training")
        logger.info(f"Training completed: {result}")

        self.save_model()
        return result

    def save_model(self, path=None):
        save_path = path or self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_path}")
        self.trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]
        logger.info("Model saved")


def train_sft(
    train_dataset,
    eval_dataset=None,
    output_dir=None,
    resume_from_checkpoint=None,
    base_model_path=None,
    config=None,
):
    """Run SFT. ``base_model_path`` points at the CPT output to continue
    from — it is load-bearing, not decorative."""
    if config:
        trainer = SFTTrainer.from_config(
            config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            base_model_path=base_model_path,
        )
    else:
        trainer = SFTTrainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            base_model_path=base_model_path,
        )
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
