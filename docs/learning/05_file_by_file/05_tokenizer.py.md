# tokenizer.py — Complete Line-by-Line Explanation

**File**: `src/baligh/models/tokenizer.py` (147 lines)
**Purpose**: Configure the Qwen2.5 tokenizer and apply chat templates for SFT formatting.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: AutoTokenizer from HF Transformers.

Line 4: Import get_model_config.

Line 5: Import logging.

Line 6: Create logger.

---

## get_tokenizer (Lines 10-47)

Lines 10-16: Function signature with parameters:
- tokenizer_name: Which tokenizer (None = Qwen2.5 default)
- max_seq_length: Maximum token count (2048)
- padding_side: Where to pad (right for decoder-only)
- truncation_side: Where to truncate (right to preserve beginning)
- add_special_tokens: Whether to add BOS/EOS tokens

Lines 17-27: Docstring.

Lines 28-30: Get default tokenizer name from ModelConfig.

Lines 31-35: Load tokenizer from HF Hub with trust_remote_code and use_fast.

Lines 37-41: Configure tokenizer settings (padding side, truncation side, max length).

Lines 43-45: Set pad_token = eos_token if missing. Qwen2.5 doesn't have a pad_token by default.

Line 47: Return configured tokenizer.

---

## get_chat_template (Lines 50-75)

Lines 50-51: Get Qwen2.5 chat template.

Lines 52-57: Docstring.

Lines 58-74: The Jinja2 chat template. It wraps messages with special tokens:
- imstart for each role (system, user, assistant)
- imend at the end of each message
- Adds generation prompt (assistant imstart) when generating

---

## apply_chat_template (Lines 78-102)

Lines 78-82: Apply chat template to messages.

Lines 83-93: Docstring.

Lines 95-96: If tokenizer has no chat template, set the Qwen2.5 template.

Lines 98-102: Call tokenizer.apply_chat_template() with messages, tokenize flag, and add_generation_prompt flag.

---

## format_instruction (Lines 105-134)

Lines 105-109: Convert instruction/input/output to message format.

Lines 110-118: Docstring.

Lines 120-127: Build user message:
- If input_text exists: combine instruction and input
- If no input: instruction alone

Lines 129-132: Add assistant message if output exists.

Line 134: Return message list.

---

## count_tokens (Lines 137-147)

Lines 137-138: Count tokens in text.

Lines 139-145: Docstring.

Line 147: Tokenize text and return length (without special tokens).
