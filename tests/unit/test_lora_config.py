"""Tests for LoRA configuration."""

from peft import LoraConfig

from baligh.training.lora_config import create_lora_config


class TestCreateLoraConfig:
    def test_returns_lora_config(self):
        config = create_lora_config()
        assert isinstance(config, LoraConfig)

    def test_default_r(self):
        config = create_lora_config()
        assert config.r == 16

    def test_default_alpha(self):
        config = create_lora_config()
        assert config.lora_alpha == 16

    def test_target_modules(self):
        config = create_lora_config()
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_custom_config(self):
        from baligh.config import LoRAConfig

        custom = LoRAConfig(r=32, lora_alpha=32)
        config = create_lora_config(custom_config=custom)
        assert config.r == 32
        assert config.lora_alpha == 32
