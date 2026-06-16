# checkpoint.py — Complete Line-by-Line Explanation

**File**: `src/baligh/training/checkpoint.py` (37 lines)
**Purpose**: Save, load, discover, and clean up training checkpoints.

---

## Imports (Lines 1-6)

Line 1: Module docstring.

Line 3: `Path` for file paths.

Line 4: `glob` for file pattern matching.

Line 5: Import logging.

Line 6: Create logger.

---

## save_checkpoint (Lines 9-15)

```python
def save_checkpoint(trainer, output_dir, step):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    trainer.save_model(str(checkpoint_dir))
    logger.info("Checkpoint saved to %s" % checkpoint_dir)
    return checkpoint_dir
```

Lines 9-15: Save a checkpoint at a specific step.
- Create directory if needed
- Save to output_dir/checkpoint-{step}
- Return the checkpoint path

---

## load_checkpoint (Lines 17-19)

```python
def load_checkpoint(trainer, checkpoint_path):
    trainer.train(resume_from_checkpoint=str(checkpoint_path))
    logger.info("Resumed from checkpoint: %s" % checkpoint_path)
```

Lines 17-19: Resume training from a checkpoint.

---

## get_latest_checkpoint (Lines 21-28)

```python
def get_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    logger.info("Latest checkpoint: %s" % latest)
    return latest
```

Lines 21-28: Find the checkpoint with the highest step number.
- Glob for checkpoint-* directories
- Sort by the numeric suffix
- Return the latest one (or None if none exist)

---

## cleanup_old_checkpoints (Lines 30-37)

```python
def cleanup_old_checkpoints(output_dir, keep_last_n=3):
    output_dir = Path(output_dir)
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if len(checkpoints) > keep_last_n:
        for cp in checkpoints[:-keep_last_n]:
            import shutil
            shutil.rmtree(cp)
            logger.info("Removed old checkpoint: %s" % cp)
```

Lines 30-37: Delete old checkpoints beyond keep_last_n.
- Sort checkpoints by step number
- Remove all but the last N
- Use shutil.rmtree for directory removal
