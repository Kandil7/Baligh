"""Training callbacks for Baligh-1.7B v0."""

from typing import Any

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from baligh.utils.logging import get_logger
from baligh.utils.memory import clear_memory, log_memory_stats

logger = get_logger(__name__)


class LoggingCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")


class MemoryCallback(TrainerCallback):
    def __init__(self, log_every_n_steps: int = 100) -> None:
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.log_every_n_steps == 0:
            log_memory_stats(prefix=f"Step {state.global_step}")
            clear_memory()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        log_memory_stats(prefix=f"Epoch {int(state.epoch or 0)}")
        clear_memory()


class CheckpointCallback(TrainerCallback):
    """Callback that triggers checkpoint saves at configured intervals."""

    def __init__(self, save_every_n_steps: int = 500) -> None:
        self.save_every_n_steps = save_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.save_every_n_steps == 0:
            control.should_save = True
