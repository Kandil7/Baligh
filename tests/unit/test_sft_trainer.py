"""Tests for SFT trainer wiring (mocked heavy deps, REAL TrainingArguments)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sft_env():
    with (
        patch("baligh.training.sft_trainer.get_tokenizer", return_value=MagicMock()) as tok,
        patch("baligh.training.sft_trainer.load_base_model") as load_model,
        patch("baligh.training.sft_trainer.apply_lora", side_effect=lambda m: m) as lora,
        patch("baligh.training.sft_trainer.resolve_precision", return_value="fp16"),
        patch("baligh.training.sft_trainer.Trainer") as trainer_cls,
    ):
        yield {
            "tokenizer": tok,
            "load_model": load_model,
            "lora": lora,
            "trainer_cls": trainer_cls,
        }


class TestSFTTrainerInit:
    def test_eval_strategy_kwarg_accepted(self, sft_env, tmp_path):
        from baligh.config import SFTConfig
        from baligh.training.sft_trainer import SFTTrainer

        t = SFTTrainer(
            config=SFTConfig(max_steps=10),
            train_dataset=[{"input_ids": [1], "labels": [1]}],
            output_dir=tmp_path / "sft",
        )
        assert t.training_args.eval_strategy == "steps"
        assert t.training_args.max_steps == 10
        assert t.training_args.num_train_epochs is None

    def test_base_model_path_is_load_bearing(self, sft_env, tmp_path):
        """Regression: --base-model used to be silently ignored, which
        disconnected the CPT->SFT pipeline entirely."""
        from baligh.config import SFTConfig
        from baligh.training.sft_trainer import SFTTrainer

        SFTTrainer(
            config=SFTConfig(),
            train_dataset=[{"input_ids": [1], "labels": [1]}],
            output_dir=tmp_path / "sft",
            base_model_path="training/cpt/final",
        )
        call_kwargs = sft_env["load_model"].call_args[1]
        assert call_kwargs["model_name"] == "training/cpt/final"

    def test_default_model_is_configured_base(self, sft_env, tmp_path):
        from baligh.config import SFTConfig, get_model_config
        from baligh.training.sft_trainer import SFTTrainer

        SFTTrainer(
            config=SFTConfig(),
            train_dataset=[{"input_ids": [1], "labels": [1]}],
            output_dir=tmp_path / "sft",
        )
        call_kwargs = sft_env["load_model"].call_args[1]
        assert call_kwargs["model_name"] == get_model_config().model_name

    def test_completion_only_loss_default(self, sft_env, tmp_path):
        from baligh.config import SFTConfig
        from baligh.training.sft_trainer import SFTTrainer

        t = SFTTrainer(
            config=SFTConfig(),
            train_dataset=[{"input_ids": [1], "labels": [1]}],
            output_dir=tmp_path / "sft",
        )
        assert t.config.response_only_loss is True
        # Seq2Seq collator pads labels with -100 so masked prompt tokens
        # never contribute to loss.
        from transformers import DataCollatorForSeq2Seq

        collator = sft_env["trainer_cls"].call_args[1]["data_collator"]
        assert isinstance(collator, DataCollatorForSeq2Seq)

    def test_signal_handler_removed_after_train(self, sft_env, tmp_path):
        from baligh.config import SFTConfig
        from baligh.training.sft_trainer import SFTTrainer

        t = SFTTrainer(
            config=SFTConfig(max_steps=1),
            train_dataset=[{"input_ids": [1], "labels": [1]}],
            output_dir=tmp_path / "sft",
        )
        mock_inner = t.trainer.return_value
        mock_inner.train.return_value = {"metrics": {}}
        t.train()
        assert t.checkpoint_manager._original_sigint is None
