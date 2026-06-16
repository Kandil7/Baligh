# formatter.py — Complete Line-by-Line Explanation

**File**: `src/baligh/data/formatter.py` (84 lines)
**Purpose**: Tokenizes and formats raw text (CPT) or instruction-response pairs (SFT) into model-ready tensors.

---

## Imports (Lines 1-7)

Line 1: Module docstring.

Line 3: Import Dataset type for type hints.

Line 4: Import tokenizer utilities:
- get_tokenizer: Load and configure the Qwen2.5 tokenizer
- apply_chat_template: Apply the chat template to messages
- format_instruction: Convert instruction/input/output to message format

Line 5: Import logging.

Line 7: Create logger bound to this module.

---

## PromptFormatter Class (Lines 9-78)

### Constructor (Lines 10-13)

Line 10: Constructor takes:
- tokenizer_name: Which tokenizer to use. None = default Qwen2.5.
- max_seq_length: Maximum token count. 2048.
- packing: Whether to pack multiple sequences into one sample.

Line 11: Load and configure the tokenizer.

Lines 12-13: Store parameters as instance attributes.

### CPT Single Example (Lines 15-21)

Line 15: Format a single CPT example (raw text).

Line 16: Extract the text field. Default to empty string if missing.

Lines 17-18: If text is empty, return empty tensors to prevent errors.

Line 19: Tokenize the text with truncation, no padding, returning Python lists.

Line 20: For CPT, labels = input_ids. The model learns to predict the next token (standard causal LM objective).

Line 21: Return the formatted example dict.

### SFT Single Example (Lines 23-37)

Line 23: Format a single SFT example (instruction/input/output).

Lines 24-26: Extract the three fields. input is optional.

Lines 27-28: If instruction or output is missing, return empty tensors.

Line 29: Convert to message format using format_instruction(). Creates user and assistant message dicts.

Line 30: Apply the Qwen2.5 chat template. This wraps messages with im_start/im_end special tokens and tokenizes the result.

Lines 31-34: Handle two cases:
- If template returns a string: tokenize it manually
- If template returns token IDs: create attention_mask as all 1s

Lines 35-36: Set labels = input_ids (same as CPT, but TRL masks user tokens internally).

Line 37: Return the formatted example.

### Chat Formatting (Lines 39-40)

Line 39: Format messages for inference (with generation prompt).

Line 40: Apply chat template with add_generation_prompt=True (adds im_start assistant at the end).

### Auto-Dispatch (Lines 42-47)

Line 42: The __call__ method makes the formatter callable.

Lines 43-44: If batch has 'instruction' key: use SFT formatting.

Lines 45-46: If batch has 'text' key: use CPT formatting.

Line 47: Fallback: return batch unchanged.

### CPT Batch Formatting (Lines 49-59)

Line 49: Format a batch of CPT examples.

Line 50: Extract the text column.

Line 51: Initialize results dict with empty lists.

Lines 52-58: For each text:
- Skip empty texts
- Tokenize with truncation
- Append input_ids, attention_mask, labels to results
- labels = copy of input_ids

Line 59: Return the batch results.

### SFT Batch Formatting (Lines 61-78)

Line 61: Format a batch of SFT examples.

Lines 62-64: Extract instruction, input, output columns. Default input to empty strings.

Line 65: Initialize results dict.

Lines 66-77: For each instruction/input/output triple:
- Skip if instruction or output is empty
- Build messages using format_instruction()
- Apply chat template (tokenize=True)
- Handle string vs token ID return
- Append to results

Line 78: Return the batch results.

---

## Factory Functions (Lines 80-84)

Line 80: Create a CPT formatter. packing=True by default for efficient pretraining.

Line 81: Return the PromptFormatter instance.

Line 83: Create an SFT formatter. packing=False by default to preserve conversation structure.

Line 84: Return the PromptFormatter instance.

---

## Design Decisions

1. **Auto-dispatch via __call__**: The formatter detects whether data is CPT or SFT based on column names, so callers don't need to know which method to call.
2. **Batch processing**: format_cpt_batch and format_sft_batch process entire batches for efficiency (tokenization is faster in batches).
3. **Empty example handling**: Returns empty tensors for missing/empty data instead of crashing.
4. **labels = input_ids**: Standard for causal LM. TRL handles response-only masking for SFT.
