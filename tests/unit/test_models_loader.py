"""Tests for model loading utilities (heavy deps mocked)."""

from unittest.mock import MagicMock, patch

import pytest


class TestBuildQuantizationConfig:
    def test_none_when_disabled(self):
        from baligh.models.loader import build_quantization_config

        assert build_quantization_config(load_in_4bit=False, load_in_8bit=False) is None

    @patch("baligh.models.loader.get_model_config")
    def test_4bit_nf4_double_quant(self, mock_cfg):
        mock_cfg.return_value = MagicMock(bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        from transformers import BitsAndBytesConfig

        from baligh.models.loader import build_quantization_config

        cfg = build_quantization_config(load_in_4bit=True, compute_dtype=None)
        assert isinstance(cfg, BitsAndBytesConfig)
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True


class TestMergeLora:
    def test_rejects_non_peft_model(self):
        """Regression: v0.1 called merge_and_unload() on a plain model —
        there was nothing to merge and it crashed confusingly."""
        from baligh.models.loader import merge_lora

        with pytest.raises(TypeError, match="no adapters to merge"):
            merge_lora(MagicMock(spec=[]), save_path=None)

    def test_merges_peft_model(self):
        from unittest.mock import patch as patch_obj

        from baligh.models.loader import merge_lora

        merged_sentinel = MagicMock()

        class FakePeftModel:
            def merge_and_unload(self):
                return merged_sentinel

        with patch_obj("baligh.models.loader.PeftModel", FakePeftModel):
            model = FakePeftModel()
            result = merge_lora(model, save_path=None)
        assert result is merged_sentinel


class TestLoadBaseModelDefaults:
    @patch("baligh.models.loader.AutoModelForCausalLM")
    @patch("baligh.models.loader.log_memory_stats", return_value={})
    def test_capability_gated_attn_on_cpu(self, _stats, mock_auto):
        """On CPU-only hosts flash_attention_2 must never be requested."""
        mock_model = MagicMock()
        mock_model.config.model_type = "qwen3"
        mock_auto.from_pretrained.return_value = mock_model

        with patch("baligh.utils.hardware.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            from baligh.models.loader import load_base_model

            load_base_model(model_name="fake/model", load_in_4bit=False)
            kwargs = mock_auto.from_pretrained.call_args[1]
            assert kwargs["attn_implementation"] == "sdpa"

    @patch("baligh.models.loader.prepare_model_for_kbit_training")
    @patch("baligh.models.loader.AutoModelForCausalLM")
    @patch("baligh.models.loader.log_memory_stats", return_value={})
    def test_inference_mode_skips_kbit_prep(self, _stats, mock_auto, mock_kbit):
        """Inference loads keep the KV cache and skip training-only prep."""
        mock_model = MagicMock()
        mock_model.config.model_type = "qwen3"
        mock_auto.from_pretrained.return_value = mock_model

        from baligh.models.loader import load_base_model

        model = load_base_model(model_name="fake/model", load_in_4bit=True, is_inference=True)
        mock_kbit.assert_not_called()
        kwargs = mock_auto.from_pretrained.call_args[1]
        assert kwargs["use_cache"] is True
        assert model is mock_model

    @patch("baligh.models.loader.prepare_model_for_kbit_training")
    @patch("baligh.models.loader.AutoModelForCausalLM")
    @patch("baligh.models.loader.log_memory_stats", return_value={})
    def test_training_mode_prepares_kbit(self, _stats, mock_auto, mock_kbit):
        mock_auto.from_pretrained.return_value = MagicMock()

        from baligh.models.loader import load_base_model

        load_base_model(model_name="fake/model", load_in_4bit=True, is_inference=False)
        mock_kbit.assert_called_once()


class TestLoadTokenizerDelegates:
    def test_loader_tokenizer_delegates_to_single_factory(self):
        """Both entry points share one implementation (v0.1 had two drifting)."""
        from unittest.mock import patch as p

        with p("baligh.models.tokenizer.AutoTokenizer") as auto:
            tok = MagicMock()
            tok.pad_token = "[PAD]"
            auto.from_pretrained.return_value = tok

            from baligh.models.loader import load_tokenizer
            from baligh.models.tokenizer import get_tokenizer

            a = load_tokenizer("some/model", 1024)
            b = get_tokenizer("some/model", 1024)
            assert a is b or type(a) is type(b)
