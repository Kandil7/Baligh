# checkpoint.py - Checkpoint Management

**Path**: `src/baligh/training/checkpoint.py` (37 lines)

## Purpose

Save, load, discover, and clean up training checkpoints.

---

## save_checkpoint (lines 9-15)

Saves trainer state to `output_dir/checkpoint-{step}`.

---

## load_checkpoint (lines 17-19)

Resumes training from a checkpoint path by calling `trainer.train(resume_from_checkpoint=...)`.

---

## get_latest_checkpoint (lines 21-28)

Finds the checkpoint with the highest step number by globbing `checkpoint-*` directories and sorting by the numeric suffix.

---

## cleanup_old_checkpoints (lines 30-37)

Deletes old checkpoints beyond `keep_last_n` (default 3). Sorts by step number, removes all but the last N using `shutil.rmtree()`.
