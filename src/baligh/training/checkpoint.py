"""Checkpoint management for Baligh-1.7B v0.

Design decisions:

1. FULL-STATE saves. ``trainer.save_model()`` alone writes weights only;
   resuming from such a directory silently restarts at step 0 with a fresh
   optimizer and LR schedule. Every manager save therefore also persists
   optimizer.pt / scheduler.pt / trainer_state.json / rng_state.pth so any
   checkpoint is resumable by HF Trainer's ``resume_from_checkpoint``.

2. SINGLE retention owner. The HF Trainer rotates its own native
   checkpoints via ``TrainingArguments(save_total_limit=...)``. This manager
   only counts/removes directories carrying our ``checkpoint_metadata.json``
   marker, so the two rotators can never evict each other's resumable state.

3. Atomic metadata writes (tmp file + os.replace) and corrupt-tolerant reads:
   validation degrades to False, never crashes the caller.

4. Interrupt handling restores the previous SIGINT disposition FIRST and
   guards against re-entrant Ctrl+C during the save itself.
"""

import json
import os
import random
import shutil
import signal
import time
from collections.abc import Callable
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


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON, returning None on any parse/read failure."""
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so a crash never leaves truncated metadata."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


class CheckpointManager:
    """Manages full-state training checkpoints with metadata and cleanup."""

    METADATA_FILE = "checkpoint_metadata.json"

    def __init__(self, output_dir: str | Path, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._original_sigint: Any = None
        self._trainer: Any = None
        self._save_callback: Callable[[], None] | None = None
        self._saving = False

    # ------------------------------------------------------------- saving

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
        """Save a FULL-STATE, resumable checkpoint plus metadata.

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
        self._save_training_state(trainer, checkpoint_dir)
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
        _write_json_atomic(checkpoint_dir / self.METADATA_FILE, asdict(metadata))

        logger.info(
            f"Checkpoint saved: {checkpoint_dir} (step={step}, loss={loss:.4f}, {elapsed:.1f}s)"
        )
        self.cleanup_old_checkpoints()
        return checkpoint_dir

    @staticmethod
    def _save_training_state(trainer: Any, checkpoint_dir: Path) -> list[str]:
        """Persist optimizer/scheduler/trainer_state/rng next to the weights.

        Each component is best-effort and independently guarded: a mock or
        partially-initialized trainer must not prevent weight saves.
        """
        saved = []
        try:
            import torch

            optimizer = getattr(trainer, "optimizer", None)
            if optimizer is not None:
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                saved.append("optimizer")

            scheduler = getattr(trainer, "lr_scheduler", None)
            if scheduler is not None:
                torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
                saved.append("scheduler")

            state = getattr(trainer, "state", None)
            if state is not None and hasattr(state, "save_to_json"):
                state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
                saved.append("trainer_state")

            rng: dict[str, Any] = {
                "python": random.getstate(),
                "numpy": __import__("numpy").random.get_state(),
                "cpu": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(rng, checkpoint_dir / "rng_state.pth")
            saved.append("rng")
        except Exception as exc:  # noqa: BLE001 - best-effort by design
            logger.warning(f"Incomplete training state saved ({saved or 'none'}): {exc}")
        return saved

    # ------------------------------------------------------------- loading

    def load_checkpoint(
        self,
        trainer: Any,
        checkpoint_path: str | Path | None = None,
    ) -> Path | None:
        """Resolve and validate a checkpoint path, then resume training.

        Args:
            trainer: HF Trainer instance.
            checkpoint_path: Specific checkpoint. If None, auto-finds latest.

        Returns:
            Path used for resumption, or None if starting fresh.
        """
        explicit = checkpoint_path is not None
        resolved: Path | None
        if explicit:
            resolved = Path(checkpoint_path)  # type: ignore[arg-type]
            if not self.validate_checkpoint(resolved):
                raise ValueError(
                    f"Requested checkpoint failed validation: {resolved}. "
                    "Refusing to silently fall back to a different one."
                )
        else:
            resolved = self.get_latest_checkpoint()
            if resolved is None:
                logger.info("No checkpoint found, starting from scratch")
                return None
            if not self.validate_checkpoint(resolved):
                logger.warning(f"Latest checkpoint invalid: {resolved}")
                fallback = self._latest_valid_checkpoint()
                if fallback is None:
                    logger.warning("No valid checkpoint found, starting from scratch")
                    return None
                resolved = fallback

        logger.info(f"Resuming from checkpoint: {resolved}")
        trainer.train(resume_from_checkpoint=str(resolved))
        return resolved

    def get_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint directory by step number."""
        checkpoints = self._list_checkpoint_dirs()
        if not checkpoints:
            return None
        latest = max(checkpoints, key=self._get_step_number)
        logger.info(f"Latest checkpoint: {latest}")
        return latest

    def _latest_valid_checkpoint(self) -> Path | None:
        """Newest checkpoint that passes validation, or None."""
        for cp in sorted(self._list_checkpoint_dirs(), key=self._get_step_number, reverse=True):
            if self.validate_checkpoint(cp):
                return cp
        return None

    # ------------------------------------------------------------ listing

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with their metadata (corrupt entries degrade)."""
        results = []
        for cp_dir in sorted(self._list_checkpoint_dirs(), key=self._get_step_number):
            info: dict[str, Any] = {"path": str(cp_dir), "step": self._get_step_number(cp_dir)}
            metadata = _read_json(cp_dir / self.METADATA_FILE)
            if metadata:
                info.update(metadata)
            results.append(info)
        return results

    def validate_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """Validate that a checkpoint exists and looks loadable.

        Checks model files, readable metadata, and step consistency. A
        corrupt/truncated metadata file fails validation instead of raising.
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
            metadata = _read_json(metadata_path)
            if metadata is None:
                logger.warning(f"Corrupt metadata in: {checkpoint_path}")
                return False
            expected_step = self._get_step_number(checkpoint_path)
            if metadata.get("step") != expected_step:
                logger.warning(
                    f"Metadata step ({metadata.get('step')}) does not match "
                    f"directory step ({expected_step})"
                )
                return False
        return True

    def get_checkpoint_metadata(self, checkpoint_path: str | Path) -> dict[str, Any] | None:
        """Read metadata from a checkpoint (None if missing/corrupt)."""
        return _read_json(Path(checkpoint_path) / self.METADATA_FILE)

    # ------------------------------------------------------------ cleanup

    def cleanup_old_checkpoints(self) -> list[Path]:
        """Remove oldest MANAGER-saved checkpoints beyond keep_last_n.

        Only directories carrying our metadata marker are considered: native
        Trainer checkpoints (no marker) are owned by
        ``TrainingArguments.save_total_limit``. This prevents two independent
        rotators from deleting each other's resumable state.
        """
        marked = [cp for cp in self._list_checkpoint_dirs() if (cp / self.METADATA_FILE).exists()]
        marked.sort(key=self._get_step_number)
        removed = []
        if len(marked) > self.keep_last_n:
            for cp in marked[: len(marked) - self.keep_last_n]:
                shutil.rmtree(cp, ignore_errors=False)
                removed.append(cp)
                logger.info(f"Removed old checkpoint: {cp}")
        return removed

    # ------------------------------------------------------------ signals

    def install_signal_handler(
        self,
        trainer: Any,
        save_fn: Callable[[], None] | None = None,
    ) -> None:
        """Install a SIGINT handler that saves a resumable checkpoint."""
        self._trainer = trainer
        self._save_callback = save_fn

        def handler(_signum: int, _frame: Any) -> None:
            if self._saving:
                # Second Ctrl+C while first save is mid-flight: do NOT nest
                # saves into the same partially-written directory.
                logger.warning("Interrupt during checkpoint save — ignoring duplicate signal")
                return
            self._saving = True
            logger.warning("Received interrupt signal — saving checkpoint before exit...")
            # Capture targets BEFORE restoring default disposition, because
            # uninstall clears them; restore first so a third Ctrl+C kills hard.
            trainer, save_fn = self._trainer, self._save_callback
            self.uninstall_signal_handler()
            try:
                if save_fn is not None:
                    save_fn()
                elif trainer is not None:
                    state = trainer.state
                    self.save_checkpoint(
                        trainer,
                        step=state.global_step,
                        epoch=state.epoch or 0.0,
                        loss=self._current_train_loss(state),
                        learning_rate=self._current_learning_rate(state),
                    )
            except Exception as exc:  # noqa: BLE001 - must not mask the interrupt
                logger.error(f"Failed to save checkpoint on interrupt: {exc}")
            finally:
                self._saving = False
                raise KeyboardInterrupt

        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handler)

    def uninstall_signal_handler(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
        self._trainer = None
        self._save_callback = None

    # -------------------------------------------------------------- intern

    @staticmethod
    def _current_train_loss(state: Any) -> float:
        """Last logged TRAINING loss (not best eval metric) for metadata."""
        try:
            history = getattr(state, "log_history", None) or []
            for entry in reversed(history):
                if isinstance(entry, dict) and "loss" in entry:
                    return float(entry["loss"])
        except Exception:  # noqa: BLE001 - metadata only
            pass
        return 0.0

    @staticmethod
    def _current_learning_rate(state: Any) -> float:
        try:
            lr = getattr(state, "learning_rate", None)
            return float(lr) if lr is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _list_checkpoint_dirs(self) -> list[Path]:
        return [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-") and d.name.split("-")[-1].isdigit()
        ]

    @staticmethod
    def _get_step_number(checkpoint_path: Path) -> int:
        try:
            return int(checkpoint_path.name.split("-")[-1])
        except (ValueError, IndexError):
            return 0
