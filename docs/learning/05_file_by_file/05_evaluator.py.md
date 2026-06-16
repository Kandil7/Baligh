# evaluator.py — Complete Line-by-Line Explanation

**File**: `src/baligh/evaluation/evaluator.py` (75 lines)
**Purpose**: Main evaluation class. Loads a model and generates responses for benchmark datasets.

---

## Imports (Lines 1-10)

Line 1: Module docstring.

Line 3: `torch` for model inference.

Line 4: `tqdm` for progress bars.

Lines 5-6: Import config and model loading functions.

Line 7: Import tokenizer utilities.

Lines 8-9: Import logging and memory utilities.

Line 10: Create logger.

---

## Evaluator Class (Lines 13-69)

### Constructor (Lines 14-27)

```python
class Evaluator:
    def __init__(self, model_path, adapter_path=None, config=None):
```

Lines 13-14: The main evaluator class.

Lines 15-16: Docstring.

Lines 17-18: Load eval config or use defaults.

Line 19: Load model config for tokenizer settings.

Line 20: Load tokenizer.

Lines 22-23: Log model loading.

Line 24: Load base model with 4-bit quantization.

Lines 26-27: If adapter_path provided, load LoRA adapter (frozen for inference).

Line 28: Set model to eval mode (disables dropout, etc.).

Line 29: Log memory usage.

### generate (Lines 29-50)

```python
    def generate(self, prompt, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None):
```

Lines 29-30: Generate a response for a single prompt.

Lines 31-34: Use provided values or fall back to config defaults.

Lines 36-37: Tokenize the prompt and move to model device.

Lines 38-48: Generate response:
- torch.no_grad(): Don't compute gradients (inference only)
- model.generate(): Run the generation loop
- Parameters: max_new_tokens, temperature, top_p, top_k, do_sample, pad_token_id, eos_token_id, repetition_penalty

Line 49: Decode only the new tokens (skip input tokens) and return response string.

### evaluate_dataset (Lines 52-69)

```python
    def evaluate_dataset(self, dataset, prompt_template=None, max_samples=None):
```

Lines 52-53: Evaluate a dataset by generating responses.

Lines 54-55: Set max_samples to dataset length if not provided.

Lines 56-68: Iterate over dataset with tqdm progress bar:
- Build prompt from template or dataset fields
- Generate response
- Collect prompt, response, reference, metadata
- Clear memory every 100 examples (prevent OOM)

Line 69: Return results list.

---

## evaluate_model Convenience Function (Lines 71-75)

```python
def evaluate_model(model_path, adapter_path=None, dataset=None, config=None):
    evaluator = Evaluator(model_path, adapter_path, config)
    if dataset:
        return evaluator.evaluate_dataset(dataset)
    return evaluator
```

Lines 71-75: Module-level convenience function. Creates Evaluator and optionally runs evaluation.
