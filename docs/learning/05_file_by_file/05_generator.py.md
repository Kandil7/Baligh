# generator.py — Complete Line-by-Line Explanation

**File**: `src/baligh/inference/generator.py` (57 lines)
**Purpose**: Core text generation class for single and batch inference.

---

## Imports (Lines 1-9)

Line 1: Module docstring.

Line 3: `torch` for inference.

Line 4: `List`, `Optional` for type hints.

Lines 5-6: Import config and model loading functions.

Line 7: Import tokenizer utilities.

Line 8: Import logging.

Line 9: Create logger.

---

## TextGenerator Class (Lines 12-53)

### Constructor (Lines 13-25)

Lines 13-14: The main generator class.

Lines 15-16: Docstring.

Lines 17-18: Load eval config.

Line 19: Load model config.

Line 20: Load tokenizer.

Lines 22-23: Log model loading.

Line 24: Load model with 4-bit quantization.

Lines 26-27: If adapter_path provided, load LoRA adapter.

Line 28: Set model to eval mode.

### generate (Lines 27-50)

```python
    def generate(self, prompt, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None, repetition_penalty=None, **kwargs):
```

Lines 27-28: Generate a response for a single prompt.

Lines 29-33: Use provided values or fall back to config defaults.

Lines 35-36: Tokenize prompt and move to device.

Lines 37-48: Generate response:
- torch.no_grad(): No gradient computation
- model.generate(): Run generation loop
- Parameters: all sampling settings
- **kwargs: Any additional generation parameters

Line 49: Decode only new tokens (skip input) and return response.

### generate_batch (Lines 52-53)

Lines 52-53: Simple sequential batch generation.

---

## generate Convenience Function (Lines 55-57)

Lines 55-57: Module-level convenience function.
