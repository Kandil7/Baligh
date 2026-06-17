"""Checkpoint management for Baligh-1.5B v0.

Provides a comprehensive CheckpointManager for saving, loading, validating,
listing, and cleaning up training checkpoints with metadata tracking.
"""

import json
import shutil
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Metadata saved alongside each checkpoint."""

    step: int
    epoch: float
    loss: float
    learning_rate: float
    global_time: str
    elapsed_seconds: float
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class CheckpointManager:
    """Manages training checkpoints with metadata, validation, and cleanup.

    Handles:
    - Saving checkpoints with metadata (step, loss, config, timestamp)
    - Loading checkpoints with validation
    - Auto-discovering the latest checkpoint
    - Listing all checkpoints with their metadata
    - Cleaning up old checkpoints
    - Signal handling for graceful interruption
    """

    METADATA_FILE = "checkpoint_metadata.json"

    def __init__(self, output_dir: str | Path, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._original_sigint = None
        self._trainer = None
        self._save_callback = None

    def save_checkpoint(
        self,
        trainer: Any,
        step: int,
        epoch: float = 0.0,
        loss: float = 0.0,
        learning_rate: float = 0.0,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save a checkpoint with metadata.

        Args:
            trainer: HF Trainer instance.
            step: Current global step.
            epoch: Current epoch.
            loss: Current training loss.
            learning_rate: Current learning rate.
            config: Training config dict.
            metrics: Additional metrics to save.

        Returns:
            Path to the saved checkpoint directory.
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        trainer.save_model(str(checkpoint_dir))
        elapsed = time.time() - start_time

        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            global_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_seconds=round(elapsed, 2),
            config=config or {},
            metrics=metrics or {},
        )

        metadata_path = checkpoint_dir / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_dir} (step={step}, loss={loss:.4f}, {elapsed:.1f}s)")

        self.cleanup_old_checkpoints()
        return checkpoint_dir

    def load_checkpoint(
        self,
        trainer: Any,
        checkpoint_path: str | Path | None = None,
    ) -> Path | None:
        """Load/resume from a checkpoint with validation.

        Args:
            trainer: HF Trainer instance.
            checkpoint_path: Specific checkpoint path. If None, auto-finds latest.

        Returns:
            Path to the loaded checkpoint, or None if no checkpoint found.
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                logger.info("No checkpoint found, starting from scratch")
                return None
        else:
            checkpoint_path = Path(checkpoint_path)

        if not self.validate_checkpoint(checkpoint_path):
            logger.warning(f"Checkpoint validation failed: {checkpoint_path}")
            logger.info("Attempting to find a valid checkpoint...")
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                logger.warning("No valid checkpoint found, starting from scratch")
                return None

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=str(checkpoint_path))
        return checkpoint_path

    def get_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint by step number.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self._list_checkpoint_dirs()
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: self._get_step_number(x))
        logger.info(f"Latest checkpoint: {latest}")
        return latest

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with their metadata.

        Returns:
            List of dicts with checkpoint info (path, step, loss, etc.).
        """
        results = []
        for cp_dir in sorted(self._list_checkpoint_dirs(), key=lambda x: self._get_step_number(x)):
            info = {"path": str(cp_dir), "step": self._get_step_number(cp_dir)}
            metadata_path = cp_dir / self.METADATA_FILE
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                info.update(metadata)
            results.append(info)
        return results

    def validate_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """Validate that a checkpoint is complete and loadable.

        Checks:
        - Directory exists
        - Contains model files (adapter_model.safetensors or model.safetensors)
        - Has metadata file
        - Metadata step matches directory name

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            True if checkpoint is valid.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_path}")
            return False

        has_model = (
            any(checkpoint_path.glob("adapter_model.*"))
            or any(checkpoint_path.glob("model.*"))
            or any(checkpoint_path.glob("*.safetensors"))
            or any(checkpoint_path.glob("*.bin"))
        )
        if not has_model:
            logger.warning(f"No model files found in: {checkpoint_path}")
            return False

        metadata_path = checkpoint_path / self.METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            expected_step = self._get_step_number(checkpoint_path)
            if metadata.get("step") != expected_step:
                logger.warning(
                    f"Metadata step ({metadata.get('step')}) does not match "
                    f"directory step ({expected_step})"
                )
                return False

        return True

    def cleanup_old_checkpoints(self) -> list[Path]:
        """Remove old checkpoints, keeping only the last N.

        Returns:
            List of removed checkpoint paths.
        """
        checkpoints = sorted(
            self._list_checkpoint_dirs(), key=lambda x: self._get_step_number(x)
        )
        removed = []
        if len(checkpoints) > self.keep_last_n:
            for cp in checkpoints[: len(checkpoints) - self.keep_last_n]:
                shutil.rmtree(cp)
                removed.append(cp)
                logger.info(f"Removed old checkpoint: {cp}")
        return removed

    def get_checkpoint_metadata(self, checkpoint_path: str | Path) -> dict[str, Any] | None:
        """Read metadata from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            Metadata dict, or None if no metadata exists.
        """
        metadata_path = Path(checkpoint_path) / self.METADATA_FILE
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            return json.load(f)

    def install_signal_handler(self, trainer: Any, save_fn=None):
        """Install SIGINT handler to save checkpoint on Ctrl+C.

        Args:
            trainer: HF Trainer instance.
            save_fn: Optional custom save function. If None, saves at current step.
        """
        self._trainer = trainer
        self._save_callback = save_fn

        def handler(_signum, _frame):
            logger.warning("Received interrupt signal — saving checkpoint before exit...")
            try:
                if self._save_callback:
                    self._save_callback()
                elif self._trainer:
                    state = self._trainer.state
                    self.save_checkpoint(
                        self._trainer,
                        step=state.global_step,
                        epoch=state.epoch or 0.0,
                        loss=state.best_metric or 0.0,
                        learning_rate=state.learning_rate,
                    )
            except Exception as e:
                logger.error(f"Failed to save checkpoint on interrupt: {e}")
            finally:
                if self._original_sigint:
                    signal.signal(signal.SIGINT, self._original_sigint)
                raise KeyboardInterrupt

        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handler)

    def uninstall_signal_handler(self):
        """Restore the original SIGINT handler."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
        self._trainer = None
        self._save_callback = None

    def _list_checkpoint_dirs(self) -> list[Path]:
        """List all checkpoint directories."""
        return [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-") and d.name.split("-")[-1].isdigit()
        ]

    def _get_step_number(self, checkpoint_path: Path) -> int:
        """Extract step number from checkpoint directory name."""
        try:
            return int(checkpoint_path.name.split("-")[-1])
        except (ValueError, IndexError):
            return 0


def save_checkpoint(trainer, output_dir, step):
    """Legacy function — use CheckpointManager.save_checkpoint() instead."""
    manager = CheckpointManager(output_dir)
    return manager.save_checkpoint(trainer, step)


def load_checkpoint(trainer, checkpoint_path):
    """Legacy function — use CheckpointManager.load_checkpoint() instead."""
    manager = CheckpointManager(Path(checkpoint_path).parent)
    return manager.load_checkpoint(trainer, checkpoint_path)


def get_latest_checkpoint(output_dir):
    """Legacy function — use CheckpointManager.get_latest_checkpoint() instead."""
    manager = CheckpointManager(output_dir)
    return manager.get_latest_checkpoint()


def cleanup_old_checkpoints(output_dir, keep_last_n=3):
    """Legacy function — use CheckpointManager.cleanup_old_checkpoints() instead."""
    manager = CheckpointManager(output_dir, keep_last_n=keep_last_n)
    return manager.cleanup_old_checkpoints()
