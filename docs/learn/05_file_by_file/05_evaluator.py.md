# evaluator.py - Main Evaluator

**Path**: `src/baligh/evaluation/evaluator.py` (75 lines)

## Purpose

Main evaluation class that loads a model and generates responses for benchmark datasets.

---

## Evaluator (lines 13-69)

### __init__ (lines 14-27)

1. Loads model with 4-bit quantization
2. Optionally loads LoRA adapter
3. Sets model to eval mode
4. Logs memory stats

### generate (lines 29-50)

Generates a response for a single prompt:
1. Tokenizes prompt
2. Calls model.generate() with configured parameters (temperature, top_p, top_k, repetition_penalty)
3. Decodes only the new tokens (skips input)
4. Returns response string

### evaluate_dataset (lines 52-69)

Iterates over a dataset, generates responses, and collects results:
- Each result contains: prompt, response, reference, metadata
- Clears memory every 100 examples
- Supports optional prompt_template for dataset-specific formatting

---

## evaluate_model (lines 71-75)

Module-level convenience function.
