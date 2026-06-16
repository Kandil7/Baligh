# model loader.py - Model Loading

**Path**: `src/baligh/models/loader.py` (251 lines)

## Purpose

Load, configure, and manage base models with quantization and LoRA adapters.

---

## load_base_model (lines 13-77)

Loads a causal LM with optional 4-bit or 8-bit quantization:
1. Creates BitsAndBytesConfig (NF4 with double quant for 4-bit)
2. Calls `AutoModelForCausalLM.from_pretrained()` with device_map="auto"
3. Prepares for k-bit training with gradient checkpointing

Parameters: model_name, load_in_4bit, load_in_8bit, device_map, torch_dtype, attn_implementation, use_cache.

---

## apply_lora (lines 80-112)

Applies LoRA adapters to a base model:
1. Gets default LoRAConfig if none provided
2. Creates PEFT LoraConfig with r=16, alpha=16, targeting all attention + MLP modules
3. Calls `get_peft_model()` to inject adapters
4. Prints trainable parameters count

---

## load_lora_model (lines 115-134)

Loads saved LoRA adapters onto a base model using `PeftModel.from_pretrained()`.

---

## merge_lora (lines 137-157)

Merges LoRA adapters into base weights using `model.merge_and_unload()`. Optionally saves the merged model with safetensors serialization.

---

## load_tokenizer (lines 160-202)

Loads and configures the tokenizer:
- Sets padding_side and truncation_side
- Sets pad_token to eos_token if missing
- Sets model_max_length

---

## prepare_model_for_training (lines 205-226)

Enables gradient checkpointing and disables KV cache.

---

## get_model_info (lines 229-251)

Returns model metadata: type, hidden_size, layers, heads, vocab_size, total/trainable params, trainable percentage.
