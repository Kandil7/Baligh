# seeding.py - Random Seeding

**Path**: `src/baligh/utils/seeding.py` (61 lines)

## Purpose

Deterministic seeding across all random libraries for reproducibility.

---

## set_seed (lines 10-46)

Sets seeds for:
1. Python random
2. NumPy
3. PyTorch CPU and CUDA
4. PYTHONHASHSEED environment variable

If config.deterministic=True:
- Sets cudnn.deterministic = True
- Disables cudnn.benchmark
- Enables deterministic algorithms (warn_only=True)

---

## seed_worker (lines 49-51)

Worker initialization function for DataLoader workers. Seeds each worker uniquely from torch.initial_seed().

---

## get_generator (lines 54-61)

Returns a torch.Generator seeded with the configured seed.
