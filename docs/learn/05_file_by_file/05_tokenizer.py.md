# tokenizer.py - Tokenizer Utilities

**Path**: `src/baligh/models/tokenizer.py` (147 lines)

## Purpose

Configure the Qwen2.5 tokenizer and apply chat templates for SFT formatting.

---

## get_tokenizer (lines 10-47)

Loads and configures the tokenizer:
1. Loads from HF with use_fast=True
2. Sets padding_side, truncation_side, model_max_length
3. Sets pad_token = eos_token if missing

---

## get_chat_template (lines 50-75)

Returns the Qwen2.5 Jinja2 chat template. Format:
- imstart for each role (system, user, assistant)
- imend at the end of each message
- Adds generation prompt (assistant imstart) when generating

---

## apply_chat_template (lines 78-102)

Applies the chat template to a list of message dicts. If the tokenizer has no chat template set, it sets the Qwen2.5 template first. Returns either a string or token IDs depending on the tokenize parameter.

---

## format_instruction (lines 105-134)

Converts instruction/input/output into message format:
- If input_text exists: combines instruction and input as user message
- If no input: instruction alone is the user message
- Output becomes the assistant message

---

## count_tokens (lines 137-147)

Simple token counting utility.
