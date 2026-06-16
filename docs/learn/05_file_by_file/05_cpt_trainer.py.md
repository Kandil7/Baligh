# cpt_trainer.py - CPT Training

**Path**: `src/baligh/training/cpt_trainer.py` (146 lines)

## Purpose

Continued Pretraining trainer. Adapts Qwen2.5-1.5B to Arabic domain by training on raw Arabic text using QLoRA.

---

## CPTTrainingState (lines 19-23)

Dataclass tracking training progress: global_step, epoch, best_loss.

---

## CPTTrainer (lines 25-139)

### __init__ (lines 26-50)

1. Loads configs (CPT, Model, Base)
2. Creates output directory
3. Sets random seed
4. Initializes tokenizer and formatter
5. Sets up model (if not provided)
6. Creates TrainingArguments
7. Creates HF Trainer

### _setup_model (lines 58-69)

1. `load_base_model()` with 4-bit quantization
2. `apply_lora()` with default LoRA config
3. `prepare_model_for_kbit_training()` with gradient checkpointing

### _create_training_args (lines 71-107)

Maps CPTConfig fields to HF TrainingArguments. Notable settings:
- `load_best_model_at_end=True`
- `gradient_checkpointing=True`
- `remove_unused_columns=False`
- Reports to wandb + tensorboard
- Uses bf16 or fp16 based on config

### _create_trainer (lines 109-119)

Creates HF `Trainer` with `DataCollatorForLanguageModeling(mlm=False)` - standard causal LM collator.

### train (lines 121-131)

Logs memory stats, calls `trainer.train()`, saves model.

### save_model (lines 133-139)

Saves model and tokenizer to output_dir/final.

---

## train_cpt (lines 141-146)

Module-level convenience function. Creates CPTTrainer and calls train().
