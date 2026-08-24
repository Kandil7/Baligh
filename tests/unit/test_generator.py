"""Tests for text generator (with mocks)."""

from unittest.mock import MagicMock, patch

from baligh.inference.generator import TextGenerator


class TestTextGenerator:
    @patch("baligh.inference.generator.load_model_with_adapter")
    @patch("baligh.inference.generator.get_tokenizer")
    def test_creation(self, mock_tok, mock_load):
        mock_load.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        gen = TextGenerator("fake_model")
        assert gen.model is not None
        mock_load.assert_called_once()

    @patch("baligh.inference.generator.load_model_with_adapter")
    @patch("baligh.inference.generator.get_tokenizer")
    def test_generate(self, mock_tok, mock_load):
        import torch

        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_load.return_value = mock_model_obj

        inputs_dict = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_obj = MagicMock()
        mock_tok_obj.return_value.to.return_value = inputs_dict
        mock_tok_obj.decode.return_value = "مرحبا بالعالم"
        mock_tok_obj.eos_token_id = 2
        mock_tok_obj.pad_token_id = 0
        mock_tok.return_value = mock_tok_obj

        gen = TextGenerator("fake_model")
        result = gen.generate("مرحبا", max_new_tokens=10)
        assert isinstance(result, str)

    @patch("baligh.inference.generator.load_model_with_adapter")
    @patch("baligh.inference.generator.get_tokenizer")
    def test_generate_zero_temperature_is_greedy(self, mock_tok, mock_load):
        """temperature=0 / do_sample=False must not sample (falsy-zero bug)."""
        import torch

        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_load.return_value = mock_model_obj

        inputs_dict = {"input_ids": torch.tensor([[1]])}
        mock_tok_obj = MagicMock()
        mock_tok_obj.return_value.to.return_value = inputs_dict
        mock_tok_obj.decode.return_value = "x"
        mock_tok_obj.eos_token_id = 2
        mock_tok_obj.pad_token_id = 0
        mock_tok.return_value = mock_tok_obj

        gen = TextGenerator("fake_model")
        gen.generate("prompt", temperature=0, do_sample=False)
        kwargs = mock_model_obj.generate.call_args[1]
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs and "top_p" not in kwargs

    @patch("baligh.inference.generator.load_model_with_adapter")
    @patch("baligh.inference.generator.get_tokenizer")
    def test_generate_batch(self, mock_tok, mock_load):
        import torch

        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_load.return_value = mock_model_obj

        inputs_dict = {"input_ids": torch.tensor([[1]])}
        mock_tok_obj = MagicMock()
        mock_tok_obj.return_value.to.return_value = inputs_dict
        mock_tok_obj.decode.return_value = "response"
        mock_tok_obj.eos_token_id = 2
        mock_tok_obj.pad_token_id = 0
        mock_tok.return_value = mock_tok_obj

        gen = TextGenerator("fake_model")
        results = gen.generate_batch(["prompt1", "prompt2"])
        assert len(results) == 2
