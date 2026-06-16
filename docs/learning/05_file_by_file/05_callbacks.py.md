# callbacks.py — Complete Line-by-Line Explanation

**File**: `src/baligh/training/callbacks.py` (33 lines)
**Purpose**: Transformer TrainerCallback subclasses for logging, memory management, and checkpointing.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: `TrainerCallback` from HF Transformers.

Line 4: Import logging.

Line 5: Import memory utilities.

Line 6: Create logger.

---

## LoggingCallback (Lines 9-12)

```python
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info("Step %d: %s" % (state.global_step, logs))
```

Lines 9-12: Log metrics at each logging event.
- on_log: Called when the trainer logs metrics
- Logs the current step and all metrics

---

## MemoryCallback (Lines 14-23)

```python
class MemoryCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            log_memory_stats(prefix="Step %d" % state.global_step)
            clear_memory()
    def on_epoch_end(self, args, state, control, **kwargs):
        log_memory_stats(prefix="Epoch %d" % int(state.epoch))
        clear_memory()
```

Lines 14-23: Monitor GPU memory and clear cache periodically.
- Every N steps: Log memory and clear CUDA cache
- At epoch end: Log memory and clear cache
- Prevents OOM errors during long training runs

---

## CheckpointCallback (Lines 25-33)

```python
class CheckpointCallback(TrainerCallback):
    def __init__(self, save_every_n_steps=500, keep_last_n=3):
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_every_n_steps == 0:
            control.should_save = True
    def on_save(self, args, state, control, **kwargs):
        pass
```

Lines 25-33: Trigger saves at configured intervals.
- Every N steps: Set should_save flag
- on_save: Placeholder for custom save logic
