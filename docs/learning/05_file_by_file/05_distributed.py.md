# distributed.py — Complete Line-by-Line Explanation

**File**: `src/baligh/utils/distributed.py` (152 lines)
**Purpose**: DDP (Distributed Data Parallel) helpers for multi-GPU training.

---

## Imports (Lines 1-8)

Line 1: Module docstring.

Line 3: `os` for environment variables.

Line 4: `torch` for PyTorch.

Line 5: `torch.distributed` for distributed training.

Line 6: Import `get_config` for settings.

Line 7: Import logging.

Line 8: Create logger.

---

## Query Functions (Lines 12-40)

### is_distributed (Lines 12-14)

```python
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()
```

Lines 12-14: Check if distributed training is active.

### get_world_size (Lines 17-21)

```python
def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1
```

Lines 17-21: Get number of processes (1 if not distributed).

### get_rank (Lines 23-28)

```python
def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0
```

Lines 23-28: Get current process rank (0 if not distributed).

### get_local_rank (Lines 30-35)

```python
def get_local_rank() -> int:
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0
```

Lines 30-35: Get local rank for multi-GPU per node (from environment variable).

### is_main_process (Lines 37-40)

```python
def is_main_process() -> bool:
    return get_rank() == 0
```

Lines 37-40: Check if current process is the main one (rank 0).

---

## setup_distributed (Lines 43-74)

```python
def setup_distributed(backend: str = "nccl") -> None:
```

Lines 43-44: Initialize distributed training.

Lines 45-50: Docstring.

Lines 52-54: If distributed not available, warn and return.

Lines 56-58: If already initialized, log and return.

Lines 60-61: Get config.

Lines 63-66: Initialize process group with NCCL backend.

Lines 68-69: Set CUDA device to local_rank.

Lines 71-74: Log the initialization.

---

## cleanup_distributed (Lines 77-80)

```python
def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
```

Lines 77-80: Clean up distributed training.

---

## barrier (Lines 84-86)

```python
def barrier() -> None:
    if is_distributed():
        dist.barrier()
```

Lines 84-86: Synchronize all processes.

---

## reduce_dict (Lines 90-117)

```python
def reduce_dict(input_dict: dict, average: bool = True) -> dict:
```

Lines 90-91: Reduce dictionary values across all processes.

Lines 92-97: Docstring.

Lines 99-100: If not distributed, return as-is.

Lines 102-103: Get world size.

Lines 104-108: Extract names and values from dict.

Lines 109-110: Stack values into tensor.

Lines 111-112: All-reduce (sum across all processes).

Lines 114-115: If averaging, divide by world size.

Lines 117: Return reduced dict.

---

## gather_object (Lines 120-134)

```python
def gather_object(obj: object) -> list:
```

Lines 120-121: Gather object from all processes.

Lines 122-128: Docstring.

Lines 130-131: If not distributed, return as single-item list.

Lines 133-134: Use all_gather_object to collect from all processes.

---

## broadcast_object (Lines 137-152)

```python
def broadcast_object(obj: object, src: int = 0) -> object:
```

Lines 137-138: Broadcast object from source to all processes.

Lines 139-145: Docstring.

Lines 147-148: If not distributed, return as-is.

Lines 150-151: Use broadcast_object_list to send from source.

Line 152: Return the broadcasted object.
