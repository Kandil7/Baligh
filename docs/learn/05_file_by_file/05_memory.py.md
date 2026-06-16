# memory.py - Memory Management

**Path**: `src/baligh/utils/memory.py` (163 lines)

## Purpose

GPU memory tracking, diagnostics, and estimation.

---

## log_memory_stats (lines 10-46)

Logs current GPU memory: allocated, reserved, peak allocated, peak reserved (all in GB).

---

## clear_memory (lines 49-54)

Runs gc.collect(), torch.cuda.empty_cache(), and torch.cuda.ipc_collect().

---

## print_model_memory (lines 57-75)

Calculates and logs parameter + buffer memory usage in GB.

---

## estimate_training_memory (lines 78-132)

Estimates total training memory from model params, batch size, seq length, precision, and optimizer:
- Model weights: params * bytes_per_param
- Gradients: same as model
- Optimizer states: 2x model (AdamW)
- Activations: batch * seq * hidden * layers * bytes * 4
- With gradient checkpointing: activations reduced by ~70%

---

## MemoryTracker (lines 135-163)

Stateful tracker that records memory snapshots with labels and computes peak/average summaries.
