"""Tests for configuration management."""

import pytest
from pathlib import Path
from baligh.config import (
    BaseConfig,
    ModelConfig,
    LoRAConfig,
    CPTConfig,
    SFTConfig,
    EvalConfig,
    QuantizationConfig,
    get_config,
    get_model_config,
    get_lora_config,
    get_cpt_config,
    get_sft_config,
    get_eval_config,
    get_quantization_config,
)


class TestBaseConfig:
    """Tests for BaseConfig."""

    def test_default_values(self):
        config = BaseConfig()
        assert config.project_name == "baligh"
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.seed == 42
        assert config.deterministic is True

    def test_path_resolution(self):
        config = BaseConfig()
        assert isinstance(config.data_dir, Path)
        assert config.data_dir.is_absolute()

    def test_factory_function(self):
        config = get_config()
        assert isinstance(config, BaseConfig)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.model_name == "Qwen/Qwen2.5-1.5B"
        assert config.max_seq_length == 2048
        assert config.load_in_4bit is True
        assert config.use_cache is False

    def test_tokenizer_name_default(self):
        config = ModelConfig()
        assert config.tokenizer_name == config.model_name

    def test_frozen_dataclass(self):
        config = ModelConfig()
        with pytest.raises(AttributeError):
            config.max_seq_length = 4096


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self):
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.0
        assert config.bias == "none"

    def test_target_modules(self):
        config = LoRAConfig()
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules
        assert "gate_proj" in config.target_modules
        assert "up_proj" in config.target_modules
        assert "down_proj" in config.target_modules


class TestCPTConfig:
    """Tests for CPTConfig."""

    def test_default_values(self):
        config = CPTConfig()
        assert config.max_steps == 50000
        assert config.learning_rate == 2e-4
        assert config.packing is True
        assert config.warmup_steps == 1000

    def test_dataset_mix(self):
        config = CPTConfig()
        assert "arabicweb24" in config.dataset_mix
        assert "arabictext_large" in config.dataset_mix
        assert "arabic_pile" in config.dataset_mix
        assert sum(config.dataset_mix.values()) == pytest.approx(1.0)


class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_default_values(self):
        config = SFTConfig()
        assert config.max_steps == 10000
        assert config.learning_rate == 1e-4
        assert config.packing is False
        assert config.response_only_loss is True
        assert config.load_best_model_at_end is True

    def test_dataset_mix(self):
        config = SFTConfig()
        assert "cidar" in config.dataset_mix
        assert "evol_instruct_arabic" in config.dataset_mix
        assert sum(config.dataset_mix.values()) == pytest.approx(1.0)


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_default_values(self):
        config = EvalConfig()
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True

    def test_human_eval_rubric(self):
        config = EvalConfig()
        assert "correctness" in config.human_eval_rubric
        assert "clarity" in config.human_eval_rubric
        assert "arabic_quality" in config.human_eval_rubric
        assert "usefulness" in config.human_eval_rubric
        assert "faithfulness" in config.human_eval_rubric


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_values(self):
        config = QuantizationConfig()
        assert config.gguf_quantization == "q4_k_m"
        assert config.awq_bits == 4
        assert config.gptq_bits == 4
        assert config.calibration_samples == 512
