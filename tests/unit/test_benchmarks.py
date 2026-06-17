"""Tests for evaluation benchmarks (with mocks)."""

from unittest.mock import MagicMock, patch

from baligh.evaluation.benchmarks import run_cidar_eval, run_islamic_qa, run_mmlu_arabic


class TestRunMmluArabic:
    @patch("baligh.evaluation.benchmarks.load_dataset")
    @patch("baligh.evaluation.benchmarks.compute_exact_match")
    def test_returns_dict(self, mock_em, mock_load):
        mock_em.return_value = 0.75
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.select.return_value = [
            {"question": "Q", "choices": {"A": "x", "B": "y"}, "answer": "A"}
        ]
        mock_load.return_value = mock_dataset

        evaluator = MagicMock()
        evaluator.generate.return_value = "A"
        result = run_mmlu_arabic(evaluator, max_samples=1)
        assert "accuracy" in result
        assert "results" in result

    @patch("baligh.evaluation.benchmarks.load_dataset")
    def test_empty_dataset(self, mock_load):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset.select.return_value = []
        mock_load.return_value = mock_dataset

        evaluator = MagicMock()
        result = run_mmlu_arabic(evaluator, max_samples=1)
        assert result["accuracy"] == 0


class TestRunCidarEval:
    @patch("baligh.evaluation.benchmarks.load_dataset")
    @patch("baligh.evaluation.benchmarks.compute_rouge")
    @patch("baligh.evaluation.benchmarks.compute_bleu")
    def test_returns_dict(self, mock_bleu, mock_rouge, mock_load):
        mock_rouge.return_value = {"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4}
        mock_bleu.return_value = 30.0
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.select.return_value = [
            {"instruction": "Q", "input": "", "output": "A"}
        ]
        mock_load.return_value = mock_dataset

        evaluator = MagicMock()
        evaluator.generate.return_value = "response"
        result = run_cidar_eval(evaluator, max_samples=1)
        assert "rouge" in result
        assert "bleu" in result


class TestRunIslamicQa:
    @patch("baligh.evaluation.benchmarks.compute_rouge")
    @patch("baligh.evaluation.benchmarks.compute_exact_match")
    def test_returns_dict(self, mock_em, mock_rouge):
        mock_rouge.return_value = {"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4}
        mock_em.return_value = 0.5
        evaluator = MagicMock()
        evaluator.generate.return_value = "response"
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=1)
        dataset.select.return_value = [{"question": "Q", "answer": "A"}]
        result = run_islamic_qa(evaluator, dataset, max_samples=1)
        assert "rouge" in result
        assert "exact_match" in result
