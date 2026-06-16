# sft_trainer.py — Complete Line-by-Line Explanation

**File**: `src/baligh/training/sft_trainer.py` (142 lines)
**Purpose**: Supervised Fine-Tuning trainer. Teaches the CPT-adapted model to follow instructions using Arabic instruction-response pairs.

---

## Imports (Lines 1-17)

Line 1: Module docstring.

Lines 3-6: Standard imports (os, dataclass, Optional, torch).

Line 7: HF Transformers: Trainer, TrainingArguments, DataCollatorForSeq2Seq.

Line 8: PEFT: LoraConfig, get_peft_model, prepare_model_for_kbit_training.

Line 9: TRL: SFTTrainer (aliased as TrlSFTTrainer), SFTConfig. TRL is the Transformer Reinforcement Learning library that provides specialized trainers for RLHF and SFT.

Lines 10-16: Project imports (config, model loading, tokenizer, formatter, logging, memory, seeding).

Line 17: Create logger.

---

## SFTTrainer Class (Lines 20-135)

### Constructor (Lines 21-45)

Lines 21-26: Same structure as CPTTrainer but uses SFTConfig. Key differences:
- Default output dir is output_dir/sft (not cpt)
- Uses get_sft_formatter() (packing=False by default)

Lines 28-29: Load model and base configs.

Lines 33-35: If no model provided, call _setup_model(). Otherwise use provided model.

Lines 41-42: Create training args and trainer.

### _setup_model (Lines 54-65)

Lines 54-65: Identical to CPTTrainer:
1. load_base_model with 4-bit quantization
2. apply_lora with default LoRA config
3. prepare_model_for_kbit_training with gradient checkpointing

### _create_training_args (Lines 67-104)

```python
    def _create_training_args(self):
        return SFTConfig(
```

Line 68: Returns TRL's SFTConfig (not HF TrainingArguments). This is a key difference from CPTTrainer.

Key SFT-specific fields:
- packing=False (line 93): Preserve conversation structure
- dataset_text_field='text' (line 94): Column name TRL reads from
- max_seq_length (line 95): From ModelConfig
- load_best_model_at_end (line 88): From SFTConfig
- metric_for_best_model='eval_loss' (line 89): Track eval_loss
- greater_is_better=False (line 90): Lower loss is better

### _create_trainer (Lines 106-115)

```python
    def _create_trainer(self):
        return TrlSFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            formatting_func=self.formatter,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors='pt'),
        )
```

Lines 106-115: Create TRL's SFTTrainer:
- formatting_func=self.formatter: The PromptFormatter that converts data to chat format
- DataCollatorForSeq2Seq: Handles padding and label masking for padded positions
- pad_to_multiple_of=8: Align sequence lengths to 8-token boundaries for GPU tensor core efficiency

### train and save_model (Lines 117-135)

Lines 117-127: Same pattern as CPTTrainer: log memory, train, log memory, save.

Lines 129-135: Same save logic as CPTTrainer.

---

## Key Differences from CPTTrainer

1. **TRL SFTTrainer** vs HF Trainer: TRL handles response-only loss automatically
2. **DataCollatorForSeq2Seq** vs DataCollatorForLanguageModeling: Handles label masking for padded positions
3. **packing=False**: Preserves conversation structure (CPT uses packing=True)
4. **Lower learning rate**: 1e-4 vs 2e-4 (SFT needs more careful updates)
5. **response_only_loss**: Loss only on assistant tokens (CPT computes loss on all tokens)

---

## train_sft Convenience Function (Lines 137-142)

Lines 137-142: Module-level convenience function. Same pattern as train_cpt but accepts base_model_path parameter.
