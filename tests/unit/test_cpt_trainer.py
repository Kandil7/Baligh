"""Tests for CPT trainer wiring (mocked heavy deps, REAL TrainingArguments)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def trainer_env():
    """Patch everything heavy; keep TrainingArguments REAL so kwarg mistakes
    (like the removed evaluation_strategy) can never pass silently again."""
    with (
        patch("baligh.training.cpt_trainer.get_tokenizer", return_value=MagicMock()) as tok,
        patch("baligh.training.cpt_trainer.load_base_model") as load_model,
        patch("baligh.training.cpt_trainer.apply_lora", side_effect=lambda m: m) as lora,
        patch("baligh.training.cpt_trainer.resolve_precision", return_value="fp16"),
        patch("baligh.training.cpt_trainer.Trainer") as trainer_cls,
    ):
        yield {
            "tokenizer": tok,
            "load_model": load_model,
            "lora": lora,
            "trainer_cls": trainer_cls,
        }


class TestCPTTrainerInit:
    def test_eval_strategy_kwarg_accepted(self, trainer_env, tmp_path):
        """Regression: evaluation_strategy was removed in transformers>=4.46;
        constructing real TrainingArguments proves eval_strategy works."""
        from baligh.config import CPTConfig
        from baligh.training.cpt_trainer import CPTTrainer

        t = CPTTrainer(
            config=CPTConfig(max_steps=10),
            train_dataset=[{"input_ids": [1]}],
            output_dir=tmp_path / "cpt",
        )
        # Real TrainingArguments object came out of _create_training_args.
        assert hasattr(t.training_args, "eval_strategy")
        assert t.training_args.eval_strategy == "steps"
        assert t.training_args.max_steps == 10

    def test_max_steps_overrides_epochs(self, trainer_env, tmp_path):
        from baligh.config import CPTConfig
        from baligh.training.cpt_trainer import CPTTrainer

        t = CPTTrainer(
            config=CPTConfig(max_steps=50),
            train_dataset=[{"input_ids": [1]}],
            output_dir=tmp_path / "cpt",
        )
        # num_train_epochs must be None when max_steps drives training.
        assert t.training_args.num_train_epochs is None

    def test_fp16_flag_set(self, trainer_env, tmp_path):
        from baligh.config import CPTConfig
        from baligh.training.cpt_trainer import CPTTrainer

        t = CPTTrainer(
            config=CPTConfig(),
            train_dataset=[{"input_ids": [1]}],
            output_dir=tmp_path / "cpt",
        )
        assert t.training_args.fp16 is True
        assert t.training_args.bf16 is False

    def test_callbacks_registered(self, trainer_env, tmp_path):
        from baligh.config import CPTConfig
        from baligh.training.cpt_trainer import CPTTrainer

        t = CPTTrainer(
            config=CPTConfig(),
            train_dataset=[{"input_ids": [1]}],
            output_dir=tmp_path / "cpt",
        )
        names = [type(c).__name__ for c in trainer_env["trainer_cls"].call_args[1]["callbacks"]]
        assert "MemoryCallback" in names and "LoggingCallback" in names

    def test_signal_handler_installed_and_removed_on_train(self, trainer_env, tmp_path):
        from baligh.config import CPTConfig
        from baligh.training.cpt_trainer import CPTTrainer

        t = CPTTrainer(
            config=CPTConfig(max_steps=1),
            train_dataset=[{"input_ids": [1]}],
            output_dir=tmp_path / "cpt",
        )
        assert t.checkpoint_manager._original_sigint is not None
        mock_inner = t.trainer.return_value
        mock_inner.train.return_value = {"metrics": {}}
        t.train()
        assert t.checkpoint_manager._original_sigint is None


class TestResolveReporters:
    def test_wandb_without_key_falls_back(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        from baligh.training.cpt_trainer import resolve_reporters

        assert resolve_reporters("proj", None) == ["tensorboard"]

    def test_wandb_with_key_enabled(self):
        from baligh.training.cpt_trainer import resolve_reporters

        assert resolve_reporters("proj", "key") == ["wandb", "tensorboard"]

    def test_offline_mode_enables_wandb(self, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "offline")
        from baligh.training.cpt_trainer import resolve_reporters

        assert resolve_reporters("proj", None) == ["wandb", "tensorboard"]
