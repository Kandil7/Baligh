# generator.py - Text Generation

**Path**: `src/baligh/inference/generator.py` (57 lines)

## Purpose

Core text generation class for single and batch inference.

---

## TextGenerator (lines 12-53)

### __init__ (lines 13-25)

Loads model with 4-bit quantization and optional LoRA adapter. Sets model to eval mode.

### generate (lines 27-50)

Single prompt generation:
1. Tokenizes prompt
2. Calls model.generate() with configurable parameters
3. Decodes only new tokens (skips input tokens)
4. Returns response string

Default parameters from EvalConfig: temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1, do_sample=True.

### generate_batch (lines 52-53)

Simple sequential batch generation (iterates generate() over prompts).

---

## generate (lines 55-57)

Module-level convenience function.
