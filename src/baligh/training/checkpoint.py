"""Checkpoint management for Baligh-1.5B v0."""

from pathlib import Path
import glob
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

def save_checkpoint(trainer, output_dir, step):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    trainer.save_model(str(checkpoint_dir))
    logger.info("Checkpoint saved to %s" % checkpoint_dir)
    return checkpoint_dir

def load_checkpoint(trainer, checkpoint_path):
    trainer.train(resume_from_checkpoint=str(checkpoint_path))
    logger.info("Resumed from checkpoint: %s" % checkpoint_path)

def get_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    logger.info("Latest checkpoint: %s" % latest)
    return latest

def cleanup_old_checkpoints(output_dir, keep_last_n=3):
    output_dir = Path(output_dir)
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if len(checkpoints) > keep_last_n:
        for cp in checkpoints[:-keep_last_n]:
            import shutil
            shutil.rmtree(cp)
            logger.info("Removed old checkpoint: %s" % cp)
