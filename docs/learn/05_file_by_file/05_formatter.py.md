# formatter.py - Prompt Formatting

**Path**: `src/baligh/data/formatter.py` (84 lines)

## Purpose

Tokenizes and formats raw text (CPT) or instruction-response pairs (SFT) into model-ready tensors.

---

## PromptFormatter (lines 9-78)

### Single-item methods

- `format_cpt(example)` (line 15): Tokenizes raw text, sets labels = input_ids
- `format_sft(example)` (line 23): Builds messages list, applies chat template, tokenizes

### Batch methods (called by __call__)

- `format_cpt_batch(batch)` (line 49): Tokenizes batch of texts, returns dict of lists
- `format_sft_batch(batch)` (line 61): Processes batch of instruction/input/output triples

### __call__ (line 42)

Auto-dispatches based on batch schema:
- Has `instruction` key: SFT formatting
- Has `text` key: CPT formatting

### format_chat (line 39)

Applies chat template for inference (with generation prompt).

---

## Factory Functions

- `get_cpt_formatter()` - PromptFormatter with packing=True
- `get_sft_formatter()` - PromptFormatter with packing=False
