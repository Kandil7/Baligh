# chat.py — Complete Line-by-Line Explanation

**File**: `src/baligh/inference/chat.py` (32 lines)
**Purpose**: Multi-turn chat interface with conversation history.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: `List`, `Dict` for type hints.

Line 4: Import `TextGenerator`.

Line 5: Import tokenizer utilities.

Line 6: Import logging.

---

## ChatBot Class (Lines 10-28)

### Constructor (Lines 11-14)

Lines 11-12: The chat interface class.

Lines 13-14: Initialize with TextGenerator, optional adapter, and system prompt.

Line 13: Default system prompt: Arabic-first assistant with Islamic knowledge.

Line 14: Initialize empty conversation history.

### chat (Lines 16-22)

```python
    def chat(self, user_message, max_new_tokens=512, temperature=0.7):
```

Lines 16-17: Send a message and get a response.

Line 18: Add user message to history.

Line 19: Build messages list with system prompt + history.

Line 20: Apply chat template (tokenize=False for string prompt).

Line 21: Generate response.

Line 22: Add assistant response to history and return it.

### clear_history (Line 24)

Line 24: Reset conversation history.

### get_history (Line 27)

Line 27: Return current conversation history.

---

## chat Convenience Function (Lines 30-32)

Lines 30-32: Module-level convenience function.
