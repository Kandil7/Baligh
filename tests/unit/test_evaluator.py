"""Tests for evaluator (with mocks)."""

from unittest.mock import MagicMock, patch

from baligh.evaluation.evaluator import Evaluator, evaluate_model


class TestEvaluator:
    @patch("baligh.evaluation.evaluator.load_base_model")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_creation(self, mock_tok, mock_model):
        mock_model.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        evaluator = Evaluator("fake_model")
        assert evaluator.model is not None

    @patch("baligh.evaluation.evaluator.load_base_model")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_generate(self, mock_tok, mock_model):
        import torch
        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = mock_model_obj

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


class TestEvaluateModel:
    @patch("baligh.evaluation.evaluator.load_base_model")
    @patch("baligh.evaluation.evaluator.get_tokenizer")
    def test_returns_evaluator(self, mock_tok, mock_model):
        mock_model.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        result = evaluate_model("fake_model")
        assert isinstance(result, Evaluator)
