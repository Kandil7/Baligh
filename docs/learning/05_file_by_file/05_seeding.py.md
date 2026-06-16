# seeding.py — Complete Line-by-Line Explanation

**File**: `src/baligh/utils/seeding.py` (61 lines)
**Purpose**: Deterministic seeding across all random libraries for reproducibility.

---

## Imports (Lines 1-8)

Line 1: Module docstring.

Line 3: `os` for environment variables.

Line 4: `random` for Python's built-in random.

Line 5: `numpy` for NumPy random.

Line 6: `torch` for PyTorch random.

Line 7: Import `get_config` for default seed.

---

## set_seed (Lines 10-46)

```python
def set_seed(seed: int | None = None) -> int:
```

Lines 10-11: Set random seed for reproducibility.

Lines 12-17: Docstring.

Lines 18-20: If no seed provided, use config default.

Lines 22-23: Set Python random seed.

Lines 25-26: Set NumPy random seed.

Lines 28-30: Set PyTorch seeds (CPU and all CUDA devices).

Lines 32-33: Set PYTHONHASHSEED environment variable (for hash reproducibility).

Lines 35-37: If deterministic mode enabled:
- cudnn.deterministic = True: Deterministic CUDA algorithms
- cudnn.benchmark = False: Disable auto-tuner (which can vary)
- torch.use_deterministic_algorithms(True, warn_only=True): Use deterministic algorithms

Lines 39-41: If not deterministic:
- cudnn.deterministic = False
- cudnn.benchmark = True: Enable auto-tuner for speed

Line 46: Return the seed that was set.

---

## seed_worker (Lines 49-51)

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Lines 49-51: Worker initialization function for DataLoader. Each worker gets a unique seed derived from torch's initial seed.

---

## get_generator (Lines 54-61)

```python
def get_generator() -> torch.Generator:
    config = get_config()
    g = torch.Generator()
    g.manual_seed(config.seed)
    return g
```

Lines 54-61: Create a torch Generator seeded with the configured seed. Used for reproducible data loading.
