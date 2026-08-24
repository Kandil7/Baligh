"""Tests for training callbacks."""

from unittest.mock import MagicMock

from baligh.training.callbacks import CheckpointCallback, LoggingCallback, MemoryCallback


class TestLoggingCallback:
    def test_on_log(self):
        cb = LoggingCallback()
        state = MagicMock()
        state.global_step = 100
        cb.on_log(args=MagicMock(), state=state, control=MagicMock(), logs={"loss": 2.0})

    def test_on_log_no_logs(self):
        cb = LoggingCallback()
        cb.on_log(args=MagicMock(), state=MagicMock(), control=MagicMock(), logs=None)


class TestMemoryCallback:
    def test_on_step_end_logs(self):
        cb = MemoryCallback(log_every_n_steps=10)
        state = MagicMock()
        state.global_step = 10
        cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())

    def test_on_step_end_skips(self):
        cb = MemoryCallback(log_every_n_steps=10)
        state = MagicMock()
        state.global_step = 5
        cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())

    def test_on_epoch_end(self):
        cb = MemoryCallback()
        state = MagicMock()
        state.epoch = 1.0
        cb.on_epoch_end(args=MagicMock(), state=state, control=MagicMock())


class TestCheckpointCallback:
    def test_triggers_save(self):
        cb = CheckpointCallback(save_every_n_steps=100)
        state = MagicMock()
        state.global_step = 100
        control = MagicMock()
        cb.on_step_end(args=MagicMock(), state=state, control=control)
        assert control.should_save is True

    def test_skips_save(self):
        cb = CheckpointCallback(save_every_n_steps=100)
        state = MagicMock()
        state.global_step = 50
        control = MagicMock()
        cb.on_step_end(args=MagicMock(), state=state, control=control)
        assert control.should_save is not True
