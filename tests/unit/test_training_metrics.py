"""Tests for training metrics."""

import math
from unittest.mock import MagicMock

import torch

from baligh.training.metrics import compute_metrics, compute_perplexity


class TestComputeMetrics:
    def test_returns_dict(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        result = compute_metrics(eval_pred=(logits, labels))
        assert isinstance(result, dict)
        assert "eval_loss" in result
        assert "perplexity" in result

    def test_perplexity_is_positive(self):
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        result = compute_metrics(eval_pred=(logits, labels))
        assert result["perplexity"] > 0


class TestComputePerplexity:
    def test_with_mock_model(self):
        model = MagicMock()
        model.eval = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(1.0)
        model.return_value = output

        dataloader = [
            {
                "input_ids": torch.randint(0, 10, (2, 5)),
                "attention_mask": torch.ones(2, 5, dtype=torch.long),
                "labels": torch.randint(0, 10, (2, 5)),
            }
        ]
        ppl = compute_perplexity(model, dataloader, "cpu")
        assert ppl > 0
        assert math.isfinite(ppl)
        model.eval.assert_called_once()
