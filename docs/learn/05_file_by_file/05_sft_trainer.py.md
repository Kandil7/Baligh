# sft_trainer.py - SFT Training

**Path**: `src/baligh/training/sft_trainer.py` (142 lines)

## Purpose

Supervised Fine-Tuning trainer. Instruction-tunes the CPT-adapted model on Arabic instruction-response pairs.

---

## SFTTrainer (lines 20-135)

### __init__ (lines 21-45)

Same structure as CPTTrainer but uses SFTConfig.

### _setup_model (lines 54-65)

Identical to CPTTrainer: load_base_model -> apply_lora -> prepare_model_for_kbit_training.

### _create_training_args (lines 67-104)

Uses TRL's `SFTConfig` (not HF TrainingArguments). Key additions:
- `packing=False` (preserve conversation structure)
- `dataset_text_field='text'`
- `max_seq_length=2048`
- `load_best_model_at_end=True`

### _create_trainer (lines 106-115)

Uses TRL's `SFTTrainer` with:
- `formatting_func=self.formatter` (auto-dispatches CPT/SFT formatting)
- `DataCollatorForSeq2Seq` with `pad_to_multiple_of=8` (GPU efficiency)

The key difference from CPTTrainer: TRL's SFTTrainer handles response-only loss automatically when using the chat template format.

### train (lines 117-127)

Same as CPTTrainer: log memory, train, save.

### save_model (lines 129-135)

Saves model and tokenizer.

---

## train_sft (lines 137-142)

Module-level convenience function.
