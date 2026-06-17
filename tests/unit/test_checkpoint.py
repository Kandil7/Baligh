"""Tests for checkpoint management."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from baligh.training.checkpoint import CheckpointManager, CheckpointMetadata


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "training" / "cpt"


@pytest.fixture
def manager(tmp_output):
    return CheckpointManager(tmp_output, keep_last_n=3)


def _create_fake_model_files(checkpoint_dir):
    """Create fake model files so validation passes."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"fake")


class TestCheckpointMetadata:
    def test_creation(self):
        meta = CheckpointMetadata(
            step=100, epoch=0.5, loss=2.5, learning_rate=0.0002,
            global_time="2026-01-01T00:00:00Z", elapsed_seconds=10.0,
            config={"lr": 0.0002}, metrics={"acc": 0.8},
        )
        assert meta.step == 100
        assert meta.loss == 2.5
        assert meta.config == {"lr": 0.0002}

    def test_frozen(self):
        meta = CheckpointMetadata(
            step=1, epoch=0.0, loss=0.0, learning_rate=0.0,
            global_time="", elapsed_seconds=0.0,
        )
        with pytest.raises(AttributeError):
            meta.step = 2


class TestCheckpointManager:
    def test_init_creates_dir(self, tmp_path):
        output = tmp_path / "nonexistent" / "dir"
        mgr = CheckpointManager(output)
        assert output.exists()

    def test_save_checkpoint(self, manager, tmp_output):
        trainer = MagicMock()
        trainer.save_model = MagicMock()

        cp_dir = manager.save_checkpoint(trainer, step=500, epoch=0.1, loss=3.0, learning_rate=0.0002)

        assert cp_dir.exists()
        assert cp_dir.name == "checkpoint-500"
        trainer.save_model.assert_called_once()

        meta_path = cp_dir / CheckpointManager.METADATA_FILE
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 500
        assert meta["loss"] == 3.0

    def test_get_latest_checkpoint(self, manager, tmp_output):
        trainer = MagicMock()
        manager.save_checkpoint(trainer, step=100, loss=5.0)
        manager.save_checkpoint(trainer, step=200, loss=4.0)
        manager.save_checkpoint(trainer, step=300, loss=3.0)

        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint-300"

    def test_get_latest_empty(self, manager):
        assert manager.get_latest_checkpoint() is None

    def test_list_checkpoints(self, manager, tmp_output):
        trainer = MagicMock()
        manager.save_checkpoint(trainer, step=100, loss=5.0)
        manager.save_checkpoint(trainer, step=200, loss=4.0)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0]["step"] == 100
        assert checkpoints[1]["step"] == 200

    def test_validate_checkpoint_valid(self, manager, tmp_output):
        trainer = MagicMock()
        cp_dir = manager.save_checkpoint(trainer, step=100, loss=5.0)
        _create_fake_model_files(cp_dir)
        assert manager.validate_checkpoint(cp_dir) is True

    def test_validate_checkpoint_missing_dir(self, manager):
        assert manager.validate_checkpoint("/nonexistent/path") is False

    def test_validate_checkpoint_no_model(self, manager, tmp_output):
        cp_dir = tmp_output / "checkpoint-100"
        cp_dir.mkdir(parents=True)
        (cp_dir / CheckpointManager.METADATA_FILE).write_text("{}")
        assert manager.validate_checkpoint(cp_dir) is False

    def test_cleanup_old_checkpoints(self, tmp_path):
        mgr = CheckpointManager(tmp_path, keep_last_n=2)
        trainer = MagicMock()

        # Create 5 checkpoints manually (bypass auto-cleanup)
        for step in [100, 200, 300, 400, 500]:
            cp_dir = tmp_path / f"checkpoint-{step}"
            _create_fake_model_files(cp_dir)
            # Write metadata manually to avoid auto-cleanup
            meta = {"step": step, "epoch": 0, "loss": 5.0, "lr": 0.0002,
                     "global_time": "", "elapsed_seconds": 0, "config": {}, "metrics": {}}
            (cp_dir / CheckpointManager.METADATA_FILE).write_text(json.dumps(meta))

        removed = mgr.cleanup_old_checkpoints()
        assert len(removed) == 3
        remaining = list(tmp_path.glob("checkpoint-*"))
        assert len(remaining) == 2

    def test_cleanup_keeps_n(self, tmp_path):
        mgr = CheckpointManager(tmp_path, keep_last_n=5)
        trainer = MagicMock()
        for step in [100, 200, 300]:
            mgr.save_checkpoint(trainer, step=step, loss=5.0)

        removed = mgr.cleanup_old_checkpoints()
        assert len(removed) == 0
        assert len(list(tmp_path.glob("checkpoint-*"))) == 3

    def test_get_checkpoint_metadata(self, manager, tmp_output):
        trainer = MagicMock()
        cp_dir = manager.save_checkpoint(trainer, step=100, loss=5.0)

        meta = manager.get_checkpoint_metadata(cp_dir)
        assert meta is not None
        assert meta["step"] == 100
        assert meta["loss"] == 5.0

    def test_get_checkpoint_metadata_none(self, manager, tmp_output):
        cp_dir = tmp_output / "checkpoint-100"
        cp_dir.mkdir(parents=True)
        assert manager.get_checkpoint_metadata(cp_dir) is None

    def test_load_checkpoint_auto(self, manager, tmp_output):
        trainer = MagicMock()
        trainer.state = MagicMock()
        trainer.state.global_step = 100
        trainer.state.epoch = 0.0
        trainer.state.best_metric = 5.0
        trainer.state.learning_rate = 0.0002

        cp_dir = manager.save_checkpoint(trainer, step=100, loss=5.0)
        _create_fake_model_files(cp_dir)

        trainer.train = MagicMock()
        result = manager.load_checkpoint(trainer)
        assert result is not None
        trainer.train.assert_called_once()

    def test_load_checkpoint_specific(self, manager, tmp_output):
        trainer = MagicMock()
        cp1 = manager.save_checkpoint(trainer, step=100, loss=5.0)
        _create_fake_model_files(cp1)
        cp2 = manager.save_checkpoint(trainer, step=200, loss=4.0)
        _create_fake_model_files(cp2)

        trainer.train = MagicMock()
        result = manager.load_checkpoint(trainer, tmp_output / "checkpoint-100")
        assert result is not None

    def test_load_checkpoint_none_found(self, manager):
        trainer = MagicMock()
        result = manager.load_checkpoint(trainer)
        assert result is None

    def test_signal_handler(self, manager):
        trainer = MagicMock()
        manager.install_signal_handler(trainer)
        assert manager._trainer is trainer
        assert manager._original_sigint is not None
        manager.uninstall_signal_handler()
        assert manager._trainer is None

    def test_signal_handler_with_callback(self, manager):
        callback = MagicMock()
        manager.install_signal_handler(MagicMock(), save_fn=callback)
        assert manager._save_callback is callback
        manager.uninstall_signal_handler()
