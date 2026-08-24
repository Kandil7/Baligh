# memory.py — Complete Line-by-Line Explanation

**File**: `src/baligh/utils/memory.py` (163 lines)
**Purpose**: GPU memory tracking, diagnostics, and estimation.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: `gc` for garbage collection.

Line 4: `torch` for CUDA memory operations.

Line 5: Import logging.

Line 6: Create logger.

---

## log_memory_stats (Lines 10-46)

```python
def log_memory_stats(device: int | torch.device | None = None, prefix: str = "") -> dict:
```

Lines 10-11: Log current GPU memory statistics.

Lines 12-17: Docstring.

Lines 18-20: If no CUDA available, return empty dict.

Lines 22-24: Get current CUDA device.

Lines 26-27: Get memory stats from CUDA.

Lines 28-31: Extract key metrics (allocated, reserved, peak allocated, peak reserved) in GB.

Lines 33-39: Log the stats with prefix.

Lines 41-46: Return stats dict.

---

## clear_memory (Lines 49-54)

```python
def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
```

Lines 49-54: Clear GPU memory cache and run garbage collection.
- gc.collect(): Run Python garbage collector
- empty_cache(): Release all unused cached memory from CUDA
- ipc_collect(): Collect inter-process memory (for multi-GPU)

---

## print_model_memory (Lines 57-75)

```python
def print_model_memory(model: torch.nn.Module) -> None:
```

Lines 57-58: Print model parameter memory usage.

Lines 60-65: Calculate parameter and buffer sizes in bytes.

Lines 67-68: Convert to GB and log.

Lines 70-75: Log the memory usage.

---

## estimate_training_memory (Lines 78-132)

```python
def estimate_training_memory(
    model_params: int,
    batch_size: int,
    seq_length: int,
    precision: str = "bf16",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> dict:
```

Lines 78-85: Estimate memory requirements for training.

Lines 86-97: Docstring.

Lines 99-100: Bytes per parameter for each precision.

Line 101: Model weights memory.

Lines 103-104: Gradients (same size as model).

Lines 106-108: Optimizer states (AdamW uses 2x model size for momentum and variance).

Lines 110-118: Activations memory:
- batch * seq * hidden * layers * bytes * 4 (factor for intermediate activations)
- For Qwen3-1.7B: hidden=2048, layers=28

Lines 120-121: If gradient checkpointing enabled, reduce activations by ~70%.

Lines 123-124: Sum all components and return.

---

## MemoryTracker Class (Lines 135-163)

Lines 135-139: Stateful tracker that records memory snapshots.

Lines 141-143: Initialize with device.

Lines 145-147: Take a memory snapshot with label.

Lines 149-163: Compute summary: peak/average allocated/reserved across all snapshots.
