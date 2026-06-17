"""Training callbacks for Baligh-1.5B v0."""

from transformers import TrainerCallback

from baligh.utils.logging import get_logger
from baligh.utils.memory import clear_memory, log_memory_stats

logger = get_logger(__name__)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info("Step %d: %s" % (state.global_step, logs))

class MemoryCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            log_memory_stats(prefix="Step %d" % state.global_step)
            clear_memory()
    def on_epoch_end(self, args, state, control, **kwargs):
        log_memory_stats(prefix="Epoch %d" % int(state.epoch))
        clear_memory()

class CheckpointCallback(TrainerCallback):
    def __init__(self, save_every_n_steps=500, keep_last_n=3):
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_every_n_steps == 0:
            control.should_save = True
    def on_save(self, args, state, control, **kwargs):
        pass
