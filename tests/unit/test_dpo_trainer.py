"""Tests for DPO trainer wiring (mocked heavy deps incl. TRL)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def dpo_env():
    with (
        patch("baligh.training.dpo_trainer.get_tokenizer", return_value=MagicMock()) as tok,
        patch("baligh.training.dpo_trainer.load_base_model") as load_model,
        patch("baligh.training.dpo_trainer.apply_lora", side_effect=lambda m: m) as lora,
        patch("baligh.training.dpo_trainer.resolve_precision", return_value="fp16"),
        patch("trl.DPOConfig") as trl_dpo_config,
        patch("trl.DPOTrainer") as trl_trainer_cls,
    ):
        yield {
            "tokenizer": tok,
            "load_model": load_model,
            "lora": lora,
            "trl_dpo_config": trl_dpo_config,
            "trl_trainer_cls": trl_trainer_cls,
        }


class TestDPOTrainerInit:
    def test_steps_mode_sets_epochs_none(self, dpo_env, tmp_path):
        from baligh.config import DPOConfig
        from baligh.training.dpo_trainer import DPOTrainer

        t = DPOTrainer(
            config=DPOConfig(max_steps=10),
            train_dataset=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            output_dir=tmp_path / "dpo",
        )
        config_kwargs = dpo_env["trl_dpo_config"].call_args[1]
        assert config_kwargs["max_steps"] == 10
        assert config_kwargs["num_train_epochs"] is None

    def test_dpo_specific_kwargs_reach_trl_config(self, dpo_env, tmp_path):
        """Regression: beta/loss_type must go inside TRL DPOConfig, not as
        bare DPOTrainer kwargs (removed in trl>=0.9 -> TypeError)."""
        from baligh.config import DPOConfig
        from baligh.training.dpo_trainer import DPOTrainer

        DPOTrainer(
            config=DPOConfig(beta=0.2, loss_type="ipo", max_length=768, max_prompt_length=256),
            train_dataset=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            output_dir=tmp_path / "dpo",
        )
        config_kwargs = dpo_env["trl_dpo_config"].call_args[1]
        assert config_kwargs["beta"] == 0.2
        assert config_kwargs["loss_type"] == "ipo"
        assert config_kwargs["max_length"] == 768
        assert config_kwargs["max_prompt_length"] == 256

    def test_trl_trainer_receives_config_as_args(self, dpo_env, tmp_path):
        from baligh.config import DPOConfig
        from baligh.training.dpo_trainer import DPOTrainer

        DPOTrainer(
            config=DPOConfig(),
            train_dataset=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            output_dir=tmp_path / "dpo",
        )
        trainer_kwargs = dpo_env["trl_trainer_cls"].call_args[1]
        assert "args" in trainer_kwargs
        assert trainer_kwargs["peft_config"] is None
        assert trainer_kwargs["ref_model"] is None

    def test_base_model_path_is_load_bearing(self, dpo_env, tmp_path):
        """Regression: --base-model must be applied BEFORE weight loading;
        setting model_name after construction silently loaded the wrong
        checkpoint (the raw base) in the first implementation."""
        from baligh.config import DPOConfig
        from baligh.training.dpo_trainer import DPOTrainer

        DPOTrainer(
            config=DPOConfig(),
            train_dataset=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            output_dir=tmp_path / "dpo",
            base_model_path="training/sft/final",
        )
        call_kwargs = dpo_env["load_model"].call_args[1]
        assert call_kwargs["model_name"] == "training/sft/final"

    def test_default_model_is_configured_base(self, dpo_env, tmp_path):
        from baligh.config import DPOConfig, get_model_config
        from baligh.training.dpo_trainer import DPOTrainer

        DPOTrainer(
            config=DPOConfig(),
            train_dataset=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
            output_dir=tmp_path / "dpo",
        )
        call_kwargs = dpo_env["load_model"].call_args[1]
        assert call_kwargs["model_name"] == get_model_config().model_name


class TestTrainDPOPlumbing:
    def test_from_config_routes_dpo_section(self, dpo_env, tmp_path):
        from baligh.training.dpo_trainer import train_dpo

        with patch.object(train_dpo.__globals__["DPOTrainer"], "train") as mock_train:
            mock_train.return_value = {"metrics": {}}
            train_dpo(
                train_dataset=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
                output_dir=tmp_path / "dpo",
                config={"dpo": {"max_steps": 5}},
            )
        config_kwargs = dpo_env["trl_dpo_config"].call_args[1]
        assert config_kwargs["max_steps"] == 5
