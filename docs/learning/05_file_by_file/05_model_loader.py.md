# model loader.py — Complete Line-by-Line Explanation

**File**: `src/baligh/models/loader.py` (251 lines)
**Purpose**: Central hub for model loading, LoRA management, merging, and inspection. Handles the full lifecycle: load quantized base model -> apply LoRA -> train -> merge LoRA -> save.

---

## Imports (Lines 1-8)

Line 1: Module docstring.

Line 3: `torch` — PyTorch.

Line 4: `AutoModelForCausalLM`, `AutoConfig` — HF model loading.

Line 5: PEFT: `LoraConfig`, `get_peft_model`, `PeftModel`, `prepare_model_for_kbit_training`.

Lines 6-8: Project imports (config, logging, memory).

---

## load_base_model (Lines 13-77)

```python
def load_base_model(
    model_name: str | None = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = "flash_attention_2",
    use_cache: bool = False,
) -> AutoModelForCausalLM:
```

Lines 13-21: Function signature with type hints and defaults. Returns an AutoModelForCausalLM.

Lines 22-35: Docstring explaining parameters and return value.

Lines 36-37: Get default model name from config if not provided.

Lines 39-40: Log and track memory before loading.

Lines 42-55: Create BitsAndBytesConfig for quantization:
- 4-bit: NF4 quantization with double quant and bfloat16 compute
- 8-bit: Simple 8-bit quantization
- Neither: Full precision (quantization_config=None)

Lines 57-66: Load the model:
- AutoModelForCausalLM.from_pretrained: Load from HF Hub
- quantization_config: BitsAndBytes config for quantization
- device_map="auto": Automatically distribute across GPUs
- torch_dtype: Compute dtype (bfloat16)
- attn_implementation: Flash Attention 2
- trust_remote_code: Required for Qwen2.5
- use_cache: Disabled for training

Lines 68-72: If quantized, prepare for k-bit training:
- prepare_model_for_kbit_training: Freezes base weights, enables gradient checkpointing

Lines 74-76: Log memory after loading and model type.

Line 77: Return the loaded model.

---

## apply_lora (Lines 80-112)

```python
def apply_lora(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig | None = None,
) -> PeftModel:
```

Lines 80-83: Function signature. Takes a base model and optional LoRA config.

Lines 84-91: Docstring.

Lines 92-106: If no LoRA config provided, create one from defaults:
- Get LoRAConfig from project config
- Map to PEFT LoraConfig with all parameters

Lines 108-109: Log the LoRA configuration.

Lines 109-110: Apply LoRA to the model:
- get_peft_model: Wraps the model with LoRA adapters
- print_trainable_parameters: Logs how many parameters are trainable

Line 112: Return the PEFT model.

---

## load_lora_model (Lines 115-134)

```python
def load_lora_model(
    base_model: AutoModelForCausalLM,
    lora_path: str,
    is_trainable: bool = False,
) -> PeftModel:
```

Lines 115-119: Load saved LoRA adapters onto a base model.

Lines 120-128: Docstring.

Lines 129-130: Log the adapter path.

Lines 131-133: Load adapters using PeftModel.from_pretrained:
- base_model: The base model to attach adapters to
- lora_path: Path to saved adapter weights
- is_trainable: False for inference, True for continued training

Line 134: Return the model with loaded adapters.

---

## merge_lora (Lines 137-157)

```python
def merge_lora(
    model: PeftModel,
    save_path: str | None = None,
) -> AutoModelForCausalLM:
```

Lines 137-140: Merge LoRA adapters into base weights.

Lines 141-148: Docstring.

Lines 149-150: Log the merge operation.

Line 151: Merge and unload:
- model.merge_and_unload(): Fuses adapter weights into base weights, removes PEFT wrapper

Lines 152-155: If save_path provided, save the merged model:
- save_pretrained: Save in safetensors format (safe_serialization=True)

Line 156: Log the save location.

Line 157: Return the merged model.

---

## load_tokenizer (Lines 160-202)

```python
def load_tokenizer(
    tokenizer_name: str | None = None,
    max_seq_length: int = 2048,
    padding_side: str = "right",
    truncation_side: str = "right",
) -> "AutoTokenizer":
```

Lines 160-165: Load and configure the tokenizer.

Lines 166-176: Docstring.

Lines 177-178: Import AutoTokenizer inside function to avoid circular imports.

Lines 179-180: Get default tokenizer name from config.

Lines 182-183: Log the tokenizer loading.

Lines 184-187: Load the tokenizer:
- AutoTokenizer.from_pretrained: Load from HF Hub
- trust_remote_code: Required for Qwen2.5
- use_fast: Use fast tokenizer implementation

Lines 189-191: Configure tokenizer:
- padding_side: Where to add padding (right for decoder-only models)
- truncation_side: Where to truncate (right to preserve beginning)

Lines 193-196: Set pad_token if missing:
- Qwen2.5 doesn't have a pad_token by default
- Set pad_token = eos_token as fallback

Lines 198-199: Set model_max_length.

Lines 201-202: Log tokenizer info (vocab size).

---

## prepare_model_for_training (Lines 205-226)

Lines 205-206: Prepare model for training.

Lines 207-216: Docstring.

Lines 218-220: Enable gradient checkpointing if requested.

Lines 223-224: Disable KV cache (not needed during training).

Line 226: Return the prepared model.

---

## get_model_info (Lines 229-251)

Lines 229-230: Get model metadata.

Lines 231-237: Docstring.

Lines 238-239: Count total parameters.

Lines 240-241: Count trainable parameters.

Lines 243-251: Return dict with:
- model_type: Model architecture type
- hidden_size: Hidden dimension
- num_layers: Number of transformer layers
- num_attention_heads: Number of attention heads
- vocab_size: Vocabulary size
- max_position_embeddings: Maximum sequence length
- total_params: Total parameter count
- trainable_params: Trainable parameter count (LoRA adapters only)
- trainable_percentage: Percentage of trainable parameters
