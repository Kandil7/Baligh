# chat.py - Chat Interface

**Path**: `src/baligh/inference/chat.py` (32 lines)

## Purpose

Multi-turn chat interface with conversation history.

---

## ChatBot (lines 10-28)

### __init__ (lines 11-14)

Creates TextGenerator and sets system prompt (default: Arabic-first assistant with Islamic knowledge).

### chat (lines 16-22)

1. Appends user message to history
2. Prepends system prompt to message list
3. Applies chat template (tokenize=False for string prompt)
4. Generates response
5. Appends assistant response to history
6. Returns response

### clear_history (line 24)

Resets conversation history.

### get_history (line 27)

Returns current conversation history.

---

## chat (lines 30-32)

Module-level convenience function.
