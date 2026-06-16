# callbacks.py - Training Callbacks

**Path**: `src/baligh/training/callbacks.py` (33 lines)

## Purpose

Transformer TrainerCallback subclasses for logging, memory management, and checkpointing.

---

## LoggingCallback (lines 9-12)

Logs metrics dictionary at each `on_log` event with the current global step.

---

## MemoryCallback (lines 14-23)

- `on_step_end`: Every `log_every_n_steps` (default 100), logs GPU memory stats and clears CUDA cache
- `on_epoch_end`: Logs memory at epoch boundaries and clears cache

Prevents OOM errors during long training runs by periodically releasing unused GPU memory.

---

## CheckpointCallback (lines 25-33)

- `on_step_end`: Sets `control.should_save = True` every `save_every_n_steps` (default 500)
- `on_save`: Currently a no-op (placeholder for custom save logic)
