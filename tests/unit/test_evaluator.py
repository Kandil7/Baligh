"""Tests for evaluator (with mocks)."""

from unittest.mock import MagicMock, patch

from baligh.evaluation.evaluator import Evaluator, evaluate_model


class TestEvaluator:
    @patch("baligh.evaluation.evaluator.load_model_with_adapter")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_creation(self, mock_tok, mock_load):
        mock_load.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        evaluator = Evaluator("fake_model")
        assert evaluator.model is not None
        mock_load.assert_called_once()

    @patch("baligh.evaluation.evaluator.load_model_with_adapter")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_creation_inference_mode(self, mock_tok, mock_load):
        """Evaluator must load in inference mode (KV cache on)."""
        mock_load.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        Evaluator("fake_model", adapter_path="adapters")
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["is_inference"] is True
        assert call_kwargs["adapter_path"] == "adapters"

    @patch("baligh.evaluation.evaluator.load_model_with_adapter")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_generate(self, mock_tok, mock_load):
        import torch

        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_load.return_value = mock_model_obj

        inputs_dict = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_obj = MagicMock()
        mock_tok_obj.return_value.to.return_value = inputs_dict
        mock_tok_obj.decode.return_value = "response"
        mock_tok_obj.eos_token_id = 2
        mock_tok_obj.pad_token_id = 0
        mock_tok.return_value = mock_tok_obj

        evaluator = Evaluator("fake_model")
        result = evaluator.generate("prompt")
        assert isinstance(result, str)

    @patch("baligh.evaluation.evaluator.load_model_with_adapter")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_generate_greedy_zero_temperature(self, mock_tok, mock_load):
        """temperature=0 must request greedy decoding, not fall back to config."""
        import torch

        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_load.return_value = mock_model_obj

        inputs_dict = {"input_ids": torch.tensor([[1, 2]])}
        mock_tok_obj = MagicMock()
        mock_tok_obj.return_value.to.return_value = inputs_dict
        mock_tok_obj.decode.return_value = "x"
        mock_tok_obj.eos_token_id = 2
        mock_tok_obj.pad_token_id = 0
        mock_tok.return_value = mock_tok_obj

        evaluator = Evaluator("fake_model")
        evaluator.generate("prompt", temperature=0, do_sample=False)
        kwargs = mock_model_obj.generate.call_args[1]
        # Greedy path must NOT pass sampling-only params.
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs


class TestEvaluateModel:
    @patch("baligh.evaluation.evaluator.load_model_with_adapter")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_returns_evaluator(self, mock_tok, mock_load):
        mock_load.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        result = evaluate_model("fake_model")
        assert isinstance(result, Evaluator)
