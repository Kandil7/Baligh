# prepare_data.py — Complete Line-by-Line Explanation

**File**: `src/scripts/prepare_data.py` (376 lines)
**Purpose**: CLI entry point that orchestrates the entire data pipeline: download, clean, deduplicate, validate, mix, format, split, and save datasets.

---

## Imports (Lines 1-33)

Line 1: Module docstring with usage examples.

Lines 3-12: Standard library imports (argparse, json, logging, os, sys, dataclass, Path, Dict, List, Optional, defaultdict).

Lines 13-14: HF datasets imports (Dataset, DatasetDict, load_dataset, concatenate_datasets, interleave_datasets).

Line 15: Add project root to sys.path.

Lines 16-17: Import project config factories.

Lines 18-22: Import data pipeline components (load datasets, cleaning, mixing, formatting, validation).

Lines 23-24: Import logging and seeding utilities.

Line 25: Create logger.

---

## DataPrepConfig (Lines 36-48)

```python
@dataclass
class DataPrepConfig:
    stage: str  # "cpt", "sft", "both", "eval"
    output_dir: Path
    clean: bool = False
    deduplicate: bool = False
    streaming: bool = True
    num_proc: int = 8
    seed: int = 42
    max_samples: Optional[int] = None
    push_to_hf: bool = False
    hf_repo: Optional[str] = None
    hf_token: Optional[str] = None
```

Lines 36-48: Configuration for data preparation.
- stage: Which data to prepare (cpt, sft, both, eval)
- output_dir: Where to save prepared data
- clean: Apply cleaning pipeline
- deduplicate: Apply MinHash LSH dedup
- streaming: Use streaming mode for large datasets
- num_proc: Parallel workers
- seed: Random seed
- max_samples: Limit samples per dataset (for testing)
- push_to_hf: Upload to HF Hub
- hf_repo: HF repo ID
- hf_token: HF authentication token

---

## DataPreparator Class (Lines 49-240)

### Constructor (Lines 51-58)

Lines 51-52: Main orchestrator class.

Lines 53-54: Store config and create output directory.

Line 56: Set seed for reproducibility.

Lines 57-58: Initialize cleaning pipeline if clean=True.

### prepare_cpt (Lines 59-108)

Lines 59-60: Prepare Continued Pretraining data.

Lines 61-63: Log CPT preparation start.

Lines 65-66: Load CPT datasets from HF.

Lines 67-70: If max_samples set, limit each dataset.

Lines 72-78: If clean=True, apply cleaning pipeline to each dataset.

Lines 80-85: If deduplicate=True, apply MinHash LSH dedup.

Lines 87-91: Validate each dataset.

Lines 93-95: Mix datasets by ratio.

Lines 97-105: Format data with tokenizer (tokenize, pack).

Lines 107-108: Split into train/eval (99/1).

### prepare_sft (Lines 109-150)

Lines 109-110: Prepare Supervised Fine-Tuning data.

Lines 111-113: Log SFT preparation start.

Lines 115-116: Load SFT datasets from HF.

Lines 117-119: If max_samples set, limit each dataset.

Lines 121-127: If clean=True, apply cleaning to instruction/input/output columns.

Lines 129-133: Validate each dataset.

Lines 135-137: Mix datasets by ratio.

Lines 139-147: Format data with chat template.

Lines 149-150: Split into train/eval (98/2).

### prepare_eval (Lines 151-160)

Lines 151-152: Prepare evaluation datasets.

Lines 153-155: Log eval preparation.

Lines 156-157: Load eval datasets.

Lines 158-160: Return as DatasetDict (no cleaning/mixing).

### Helper Methods (Lines 161-240)

Lines 161-174: _clean_dataset: Apply cleaning pipeline to text column.

Lines 175-189: _clean_sft_dataset: Apply cleaning to instruction/input/output.

Lines 190-209: _split_dataset: Split into train/eval.

Lines 210-230: save: Save to disk with metadata.json.

Lines 231-240: push_to_hf: Push to HF Hub.

---

## CLI Arguments (Lines 241-332)

Lines 241-332: Parse command-line arguments.
- --stage: cpt, sft, both, eval
- --output-dir: Output location
- --clean: Apply cleaning
- --deduplicate: Apply dedup
- --streaming: Use streaming
- --num-proc: Parallel workers
- --seed: Random seed
- --max-samples: Limit for testing
- --push-to-hf: Upload to Hub
- --hf-repo: Hub repo ID
- --hf-token: Auth token
- --log-level: Logging verbosity

---

## main (Lines 333-376)

Lines 333-376: Main entry point.
- Parse arguments
- Setup logging
- Get HF token from env if not provided
- Create DataPreparator
- Run preparation for requested stages
- Save results
- Optionally push to HF Hub
- Handle errors gracefully
