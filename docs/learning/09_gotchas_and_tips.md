# 09 — Gotchas and Tips: Baligh-1.7B v0

## Common Gotchas

### 1. Streaming vs Non-Streaming Dataset Access

**Gotcha**: Streaming datasets don't support `len()`, `shuffle()`, or random access. Calling `dataset[i]` on a streaming dataset raises an error.

**Tip**: Use `ds.take(n)` and `ds.skip(n)` for streaming. For `train_test_split`, use `take(eval_size)` for eval and `skip(eval_size)` for train. Only use non-streaming for small datasets that fit in memory.

### 2. Packing Breaks Conversation Structure

**Gotcha**: When `packing=True` in SFT, multiple conversations get concatenated into one sequence. This corrupts the chat template boundaries and the response-only loss computation.

**Tip**: Always set `packing=False` for SFT. Packing is safe for CPT where sequences are raw text with no structure.

### 3. pad_token Missing

**Gotcha**: Qwen2.5 tokenizer doesn't have a pad_token by default. This causes errors when creating batches with padding.

**Tip**: The codebase automatically sets `pad_token = eos_token` in `get_tokenizer()` and `load_tokenizer()`. If you create a tokenizer outside these functions, set it manually.

### 4. Labels Are input_ids (CPT) or Full Sequence (SFT)

**Gotcha**: For CPT, `labels = input_ids` (predict every token). For SFT with response-only loss, `labels` also equals `input_ids`, but TRL's SFTTrainer masks user tokens internally using the chat template.

**Tip**: Don't manually mask labels in SFT. Let TRL handle it via the `formatting_func` and chat template.

### 5. 4-bit Quantization Changes Model Behavior

**Gotcha**: Loading a model in 4-bit NF4 produces different outputs than loading in FP16. Quantization introduces small errors.

**Tip**: For evaluation, consider merging LoRA adapters first, then loading the merged model in FP16 for accurate benchmarking. Only use 4-bit for training to save VRAM.

### 6. Flash Attention 2 Requires Specific Hardware

**Gotcha**: Flash Attention 2 requires Ampere (A100) or newer GPUs. Older GPUs (V100, T4) will fall back to SDPA or standard attention.

**Tip**: The code defaults to `flash_attention_2` but will error on incompatible hardware. Set `attn_implementation="sdpa"` for older GPUs.

### 7. interleave_datasets Stopping Strategy

**Gotcha**: `stopping_strategy='first_exhausted'` means training stops when the smallest dataset (ArabicText-Large at ~1B tokens) runs out, even though ArabicWeb24 has 28B tokens.

**Tip**: This is by design to prevent one dataset from dominating. If you need more data, add more datasets or use `stopping_strategy='all_exhausted'` (which repeats smaller datasets).

### 8. Frozen Configs Cannot Be Modified

**Gotcha**: All config dataclasses are frozen. `config.learning_rate = 1e-5` raises `FrozenInstanceError`.

**Tip**: Create a new config with overridden values: `CPTConfig(learning_rate=1e-5)` or pass overrides via YAML config files.

### 9. merge_lora Requires Full-Precision Loading

**Gotcha**: You cannot merge LoRA adapters into a 4-bit quantized model. The merge operation requires full-precision weights.

**Tip**: Load the base model with `load_in_4bit=False` before calling `merge_lora()`. The `merge_lora.py` script handles this correctly.

### 10. Response-Only Loss Needs Chat Template

**Gotcha**: TRL's SFTTrainer only applies response-only masking when the data uses a recognized chat template format. Raw instruction/output pairs without template formatting will compute loss on all tokens.

**Tip**: Always use `format_instruction()` to build message dicts, then `apply_chat_template()` to format them. The formatter handles this automatically.

## Performance Tips

### 1. Use Streaming for Large Datasets

```python
# Good: Streams from HF Hub, no download needed
ds = load_dataset("lightonai/ArabicWeb24", split="train", streaming=True)

# Bad: Downloads 28GB+ to disk
ds = load_dataset("lightonai/ArabicWeb24", split="train", streaming=False)
```

### 2. Enable Gradient Checkpointing

Gradient checkpointing reduces activation memory by ~70% with ~30% compute overhead. Always enable for QLoRA training:

```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
```

### 3. Use Fused AdamW

The fused AdamW optimizer is 15-20% faster on CUDA than the standard implementation:

```python
TrainingArguments(optim="adamw_torch_fused")
```

### 4. Pad to Multiple of 8

For SFT, `pad_to_multiple_of=8` in the DataCollator aligns sequences to GPU tensor core dimensions:

```python
DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
```

### 5. Clear Memory Periodically

The `MemoryCallback` clears CUDA cache every 100 steps. For custom training loops, call `clear_memory()` regularly:

```python
from baligh.utils.memory import clear_memory

clear_memory()  # gc.collect() + cuda.empty_cache()
```

### 6. Use bf16 Over fp16

BF16 has a larger dynamic range than FP16, making it more stable for training. Use it when hardware supports it (Ampere+):

```python
TrainingArguments(bf16=True, fp16=False)
```

### 7. Limit save_total_limit

Keep only the last N checkpoints to save disk space:

```python
TrainingArguments(save_total_limit=3)
```

### 8. Use load_best_model_at_end for SFT

Automatically loads the checkpoint with the best eval_loss at the end of training:

```python
SFTConfig(load_best_model_at_end=True, metric_for_best_model="eval_loss")
```

## Debugging Tips

### 1. Test with max_samples

```bash
python -m src.scripts.prepare_data --stage cpt --clean --max-samples 100
```

### 2. Check Dataset Statistics

```python
from baligh.data.validators import get_dataset_stats

stats = get_dataset_stats(dataset, text_column="text")
print(stats)  # {count, mean_length, median_length, min_length, max_length, total_chars}
```

### 3. Log Memory at Any Point

```python
from baligh.utils.memory import log_memory_stats

log_memory_stats(prefix="Debug checkpoint")
```

### 4. Check Trainable Parameters

```python
model = apply_lora(model)  # Prints trainable parameter count
# Or: model.print_trainable_parameters()
```

### 5. Resume from Checkpoint

```bash
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --resume training/cpt/checkpoint-5000
```

### 6. Inspect Model Info

```python
from baligh.models.loader import get_model_info

info = get_model_info(model)
print(info)  # {model_type, hidden_size, num_layers, total_params, trainable_params, ...}
```

## Deployment Tips

### 1. GGUF for Local Deployment

GGUF is the best format for local deployment with llama.cpp/Ollama:

```bash
python -m src.scripts.quantize --model-path merged_model --output-dir ./gguf --method gguf --quantization q4_k_m
```

### 2. AWQ for vLLM

AWQ provides the best speed/quality trade-off for vLLM serving:

```bash
python -m src.scripts.quantize --model-path merged_model --output-dir ./awq --method awq --bits 4
```

### 3. Push to Hugging Face

```bash
python -m src.scripts.push_to-hf --model-path ./gguf --repo-id Kandil7/Baligh-1.7B --token $HF_TOKEN
```

### 4. Generate Model Card

```bash
python -m src.scripts.generate_model_card --output RELEASE.md --mmlu-score 65.2 --cidar-rouge 42.1
```
