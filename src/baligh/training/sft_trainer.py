"""Supervised Fine-Tuning Trainer for Baligh-1.5B v0."""

import torch
from peft import prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig  # type: ignore[import-untyped]
from trl import SFTTrainer as TrlSFTTrainer  # type: ignore[import-untyped]

from baligh.config import get_config, get_model_config, get_sft_config
from baligh.data.formatter import get_sft_formatter
from baligh.models.loader import apply_lora, load_base_model
from baligh.models.tokenizer import get_tokenizer
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats
from baligh.utils.seeding import set_seed

logger = get_logger(__name__)

class SFTTrainer:
    def __init__(self, config=None, model=None, tokenizer=None, train_dataset=None, eval_dataset=None, output_dir=None):
        self.config = config or get_sft_config()
        self.model_config = get_model_config()
        self.base_config = get_config()
        self.output_dir = output_dir or self.base_config.output_dir / 'sft'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.base_config.seed)

        self.tokenizer = tokenizer or get_tokenizer(self.model_config.tokenizer_name, self.model_config.max_seq_length)
        self.formatter = get_sft_formatter(self.model_config.tokenizer_name, self.model_config.max_seq_length)

        if model is None:
            self.model = self._setup_model()
        else:
            self.model = model

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.training_args = self._create_training_args()
        self.trainer = self._create_trainer()

        logger.info('SFTTrainer initialized')
        log_memory_stats(prefix='After trainer init')

    @classmethod
    def from_config(cls, config_dict: dict, **kwargs):
        """Create trainer from config dictionary."""
        from baligh.config import SFTConfig
        config = SFTConfig(**config_dict.get("sft", {}))
        return cls(config=config, **kwargs)

    def _setup_model(self):
        model = load_base_model(
            model_name=self.model_config.unsloth_model_name,
            load_in_4bit=self.model_config.load_in_4bit,
            load_in_8bit=self.model_config.load_in_8bit,
            torch_dtype=torch.bfloat16 if self.model_config.bnb_4bit_compute_dtype == 'bfloat16' else torch.float16,
            attn_implementation=self.model_config.attn_implementation,
            use_cache=False,
        )
        model = apply_lora(model)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return model

    def _create_training_args(self):
        return SFTConfig(
            output_dir=str(self.output_dir),
            max_steps=self.config.max_steps,
            num_train_epochs=self.config.num_train_epochs,
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
            evaluation_strategy=self.config.evaluation_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            packing=self.config.packing,
            dataset_text_field='text',
            max_seq_length=self.model_config.max_seq_length,
            report_to=['wandb', 'tensorboard'] if self.base_config.wandb_project else ['tensorboard'],
            run_name='baligh-sft',
            seed=self.base_config.seed,
            data_seed=self.base_config.seed,
            bf16=self.base_config.mixed_precision == 'bf16',
            fp16=self.base_config.mixed_precision == 'fp16',
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
        )

    def _create_trainer(self):
        return TrlSFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,  # type: ignore[call-arg]
            formatting_func=self.formatter,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors='pt'),
        )

    def train(self, resume_from_checkpoint=None):
        logger.info('Starting SFT training')
        log_memory_stats(prefix='Before training')

        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        log_memory_stats(prefix='After training')
        logger.info('Training completed: %s' % result)

        self.save_model()
        return result

    def save_model(self, path=None):
        save_path = path or self.output_dir / 'final'
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info('Saving model to %s' % save_path)
        self.trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]
        logger.info('Model saved')

def train_sft(train_dataset, eval_dataset=None, output_dir=None, resume_from_checkpoint=None, base_model_path=None, config=None):
    if config:
        trainer = SFTTrainer.from_config(config, train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=output_dir)
    else:
        trainer = SFTTrainer(train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=output_dir)
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
