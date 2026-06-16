# distributed.py - Distributed Training

**Path**: `src/baligh/utils/distributed.py` (152 lines)

## Purpose

DDP (Distributed Data Parallel) helpers for multi-GPU training.

---

## Query Functions

- `is_distributed()` (line 12): Checks if dist is available and initialized
- `get_world_size()` (line 17): Returns number of processes
- `get_rank()` (line 23): Returns current process rank
- `get_local_rank()` (line 30): Returns LOCAL_RANK from env
- `is_main_process()` (line 37): Checks if rank == 0

---

## setup_distributed (lines 43-74)

Initializes NCCL process group with env:// init method. Sets CUDA device to local_rank.

---

## cleanup_distributed (lines 77-80)

Destroys the process group.

---

## barrier (line 84)

Synchronizes all processes.

---

## reduce_dict (lines 90-117)

Reduces a dictionary of tensors across all processes (all_reduce). Optionally averages.

---

## gather_object (lines 120-134)

Gathers Python objects from all processes using all_gather_object.

---

## broadcast_object (lines 137-152)

Broadcasts a Python object from source rank to all processes.
