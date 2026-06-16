# quantization.py — Complete Line-by-Line Explanation

**File**: `src/baligh/models/quantization.py` (147 lines)
**Purpose**: Export merged models to quantized formats (GGUF, AWQ, GPTQ) for efficient inference deployment.

---

## Imports (Lines 1-7)

Line 1: Module docstring.

Line 3: `dataclass` for config classes.

Line 4: Import `get_quantization_config` for default settings.

Line 5: Import logging.

Line 7: Create logger.

---

## Config Dataclasses (Lines 10-33)

### GGUFConfig (Lines 10-14)

```python
@dataclass(frozen=True, slots=True)
class GGUFConfig:
    quantization: str = "q4_k_m"
    output_dir: str = "./gguf"
```

Lines 10-14: GGUF quantization settings.
- quantization: "q4_k_m" = 4-bit with k-quant medium quality. Options: q4_k_m, q5_k_m, q8_0, f16, f32.
- output_dir: Where to save the GGUF file.

### AWQConfig (Lines 17-24)

```python
@dataclass(frozen=True, slots=True)
class AWQConfig:
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"
    output_dir: str = "./awq"
```

Lines 17-24: AWQ (Activation-aware Weight Quantization) settings.
- bits: 4-bit quantization.
- group_size: Quantize groups of 128 weights together.
- zero_point: Use zero point for asymmetric quantization.
- version: "gemm" = use GEMM kernel for fast matrix multiplication.

### GPTQConfig (Lines 27-33)

```python
@dataclass(frozen=True, slots=True)
class GPTQConfig:
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    output_dir: str = "./gptq"
```

Lines 27-33: GPTQ settings.
- bits: 4-bit quantization.
- group_size: Quantize groups of 128 weights.
- desc_act: Quantize activations in descending order of importance.

---

## quantize_gguf (Lines 36-69)

```python
def quantize_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
) -> None:
```

Lines 36-40: Convert model to GGUF format using llama.cpp.

Lines 41-47: Docstring.

Lines 48-49: Import subprocess and os (for running external commands).

Lines 51-52: Log the quantization.

Lines 54-60: Build the conversion command:
- python -m llama_cpp.convert: Use llama.cpp's conversion script
- --model: Input model path
- --outfile: Output GGUF file
- --outtype: Quantization type

Lines 62-63: Log the command.

Lines 64-65: Run the command as a subprocess.

Lines 67-69: Check for errors and raise if conversion failed.

Line 69: Log success.

---

## quantize_awq (Lines 72-109)

```python
def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "gemm",
) -> None:
```

Lines 72-79: Quantize model using AWQ.

Lines 80-89: Docstring.

Lines 90-91: Import autoawq and AutoTokenizer.

Lines 93-94: Log the quantization.

Lines 95-96: Load model and tokenizer with autoawq.

Lines 98-103: Create AWQ quantization config.

Lines 105-107: Quantize and save:
- model.quantize: Run AWQ quantization
- save_quantized: Save quantized model
- save_pretrained: Save tokenizer

Line 109: Log success.

---

## quantize_gptq (Lines 112-147)

```python
def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
) -> None:
```

Lines 112-118: Quantize model using GPTQ.

Lines 119-126: Docstring.

Lines 127-128: Import auto_gptq and AutoTokenizer.

Lines 130-131: Log the quantization.

Lines 133-134: Load tokenizer.

Lines 136-142: Load model with quantize_config.

Lines 144-145: Save quantized model and tokenizer.

Line 147: Log success.
