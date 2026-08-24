# cpt_trainer.py — Complete Line-by-Line Explanation

**File**: `src/baligh/training/cpt_trainer.py` (146 lines)
**Purpose**: Continued Pretraining trainer. Adapts Qwen3-1.7B to Arabic domain by training on raw Arabic text using QLoRA.

---

## Imports (Lines 1-16)

Line 1: Module docstring.

Line 3: `os` for environment variables.

Line 4: `dataclass` for CPTTrainingState.

Line 5: `Optional` for type hints.

Line 6: `torch` — PyTorch deep learning framework.

Line 7: `Trainer`, `TrainingArguments`, `DataCollatorForLanguageModeling` from Hugging Face Transformers. These are the core training components.

Line 8: `LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training` from PEFT. These handle LoRA adapter injection and quantized training preparation.

Line 9: Import project config factories.

Line 10: Import model loading functions: `load_base_model` (loads 4-bit model), `apply_lora` (injects LoRA adapters).

Line 11: Import `get_tokenizer` for tokenizer setup.

Line 12: Import `get_cpt_formatter` for CPT data formatting.

Line 13: Import logging.

Line 14: Import memory utilities: `log_memory_stats` (logs GPU memory), `clear_memory` (frees GPU cache).

Line 15: Import `set_seed` for reproducibility.

Line 16: Create logger.

---

## CPTTrainingState (Lines 19-23)

```python
@dataclass
class CPTTrainingState:
    global_step: int = 0
    epoch: float = 0.0
    best_loss: float = float("inf")
```

Lines 19-23: Simple dataclass to track training progress.
- global_step: Current training step (incremented each batch)
- epoch: Current epoch (fractional, e.g., 2.5 = halfway through epoch 3)
- best_loss: Best validation loss seen (starts at infinity so any real loss is lower)

---

## CPTTrainer Class (Lines 25-139)

### Constructor (Lines 26-50)

```python
class CPTTrainer:
    def __init__(self, config=None, model=None, tokenizer=None, train_dataset=None, eval_dataset=None, output_dir=None):
```

Lines 25-26: The main trainer class. All parameters are optional with defaults.

```python
        self.config = config or get_cpt_config()
```
Line 27: Use provided config or create default CPTConfig.

```python
        self.model_config = get_model_config()
        self.base_config = get_base_config()
```
Lines 28-29: Always load model and base configs (they provide hardware/logging settings).

```python
self.output_dir = output_dir or self.base_config.output_dir / "cpt"
self.output_dir.mkdir(parents=True, exist_ok=True)
```
Lines 30-31: Set output directory. Create it if it doesn't exist.

```python
        set_seed(self.base_config.seed)
```
Line 33: Set random seed for reproducibility.

```python
self.tokenizer = tokenizer or get_tokenizer(
    self.model_config.tokenizer_name, self.model_config.max_seq_length
)
```
Line 35: Load tokenizer if not provided.

```python
self.formatter = get_cpt_formatter(
    self.model_config.tokenizer_name, self.model_config.max_seq_length, packing=self.config.packing
)
```
Line 36: Create CPT formatter with packing setting from config.

```python
        if model is None:
            self.model = self._setup_model()
        else:
            self.model = model
```
Lines 38-41: If no model provided, set up a new one. Otherwise, use the provided model.

```python
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
```
Lines 43-44: Store datasets.

```python
        self.training_args = self._create_training_args()
        self.trainer = self._create_trainer()
```
Lines 46-47: Create training arguments and HF Trainer instance.

```python
logger.info("CPTTrainer initialized")
log_memory_stats(prefix="After trainer init")
```
Lines 49-50: Log initialization and GPU memory usage.

### from_config Class Method (Lines 51-56)

```python
    @classmethod
    def from_config(cls, config_dict: dict, **kwargs):
```
Line 52: Alternative constructor. Takes a raw dictionary (e.g., from YAML file) instead of a CPTConfig object.

```python
from baligh.config import CPTConfig

config = CPTConfig(**config_dict.get("cpt", {}))
return cls(config=config, **kwargs)
```
Lines 54-56: Extract the "cpt" key from the dict, create a CPTConfig, pass to normal constructor.

### _setup_model (Lines 58-69)

```python
    def _setup_model(self):
```
Line 58: Core model assembly pipeline.

```python
model = load_base_model(
    model_name=self.model_config.unsloth_model_name,
    load_in_4bit=self.model_config.load_in_4bit,
    load_in_8bit=self.model_config.load_in_8bit,
    torch_dtype=torch.bfloat16
    if self.model_config.bnb_4bit_compute_dtype == "bfloat16"
    else torch.float16,
    attn_implementation=self.model_config.attn_implementation,
    use_cache=False,
)
```
Lines 59-66: Load the base model with:
- Unsloth 4-bit model (pre-quantized for faster loading)
- 4-bit quantization enabled
- bfloat16 compute dtype (for training stability)
- Flash Attention 2 (for memory efficiency)
- KV cache disabled (not needed during training)

```python
        model = apply_lora(model)
```
Line 67: Inject LoRA adapters into the model. This wraps the model with PEFT.

```python
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
```
Line 68: Prepare the model for quantized training:
- Freeze base weights (only LoRA adapters are trainable)
- Enable gradient checkpointing (trades compute for memory)

```python
        return model
```
Line 69: Return the prepared model.

### _create_training_args (Lines 71-107)

```python
    def _create_training_args(self):
```
Line 71: Create HF TrainingArguments from config.

```python
return TrainingArguments(
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
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=self.config.dataloader_num_workers,
    dataloader_pin_memory=self.config.dataloader_pin_memory,
    dataloader_drop_last=self.config.dataloader_drop_last,
    remove_unused_columns=False,
    report_to=["wandb", "tensorboard"] if self.base_config.wandb_project else ["tensorboard"],
    run_name="baligh-cpt",
    seed=self.base_config.seed,
    data_seed=self.base_config.seed,
    bf16=self.base_config.mixed_precision == "bf16",
    fp16=self.base_config.mixed_precision == "fp16",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)
```

Lines 72-107: Map every config field to a TrainingArguments field. Key settings:
- load_best_model_at_end=True: Keep the checkpoint with lowest eval_loss
- metric_for_best_model='eval_loss': Which metric determines "best"
- greater_is_better=False: Lower eval_loss is better
- remove_unused_columns=False: Don't let Trainer strip columns the data collator needs
- report_to: Log to W&B if configured, otherwise just TensorBoard
- bf16/fp16: Mixed precision based on config
- gradient_checkpointing=True: Save memory by recomputing activations
- ddp_find_unused_parameters=False: Optimize DDP by assuming all parameters are used

### _create_trainer (Lines 109-119)

```python
    def _create_trainer(self):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
```

Lines 109-119: Create the HF Trainer:
- DataCollatorForLanguageModeling: Pads batches and creates labels
- mlm=False: Causal LM (predict next token), not masked LM (BERT-style)
- Trainer: The main training loop implementation

### train (Lines 121-131)

```python
def train(self, resume_from_checkpoint=None):
    logger.info("Starting CPT training")
    log_memory_stats(prefix="Before training")

    result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    log_memory_stats(prefix="After training")
    logger.info("Training completed: %s" % result)

    self.save_model()
    return result
```

Lines 121-131: The training loop:
1. Log memory before training
2. Call trainer.train() (this runs the actual training loop)
3. Log memory after training
4. Save the final model
5. Return the training results

### save_model (Lines 133-139)

```python
def save_model(self, path=None):
    save_path = path or self.output_dir / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s" % save_path)
    self.trainer.save_model(str(save_path))
    self.tokenizer.save_pretrained(str(save_path))
    logger.info("Model saved")
```

Lines 133-139: Save the model and tokenizer:
- Default save location: output_dir/final
- trainer.save_model: Saves model weights and config
- tokenizer.save_pretrained: Saves tokenizer files (vocab, merges, etc.)

---

## train_cpt Convenience Function (Lines 141-146)

```python
def train_cpt(
    train_dataset, eval_dataset=None, output_dir=None, resume_from_checkpoint=None, config=None
):
    if config:
        trainer = CPTTrainer.from_config(
            config, train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=output_dir
        )
    else:
        trainer = CPTTrainer(
            train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=output_dir
        )
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

Lines 141-146: Module-level convenience function. Creates a CPTTrainer and calls train(). Supports both dict config (from YAML) and default config.
