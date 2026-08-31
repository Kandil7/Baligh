"""Tests for evaluation benchmarks (with mocks)."""

import json
from unittest.mock import MagicMock, patch

from baligh.evaluation.benchmarks import (
    _check_citation,
    _check_refusal,
    run_cidar_eval,
    run_islamic_qa,
    run_mmlu_arabic,
    run_sunni_benchmark,
)


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
        mock_dataset.select.return_value = [{"instruction": "Q", "input": "", "output": "A"}]
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


class TestSunniCheckers:
    def test_citation_pass_on_matching_response(self):
        item = {"required_citation": {"surah": 2, "ayah": 255}}
        response = "آية الكرسي هي الآية 255 من سورة البقرة"
        assert _check_citation(item, response) is True

    def test_citation_fails_on_missing_number(self):
        item = {"required_citation": {"collection": "صحيح البخاري", "number": 1}}
        fabricated = "رواه مسلم رقم 1900"
        assert _check_citation(item, fabricated) is False

    def test_citation_accepts_eastern_digits(self):
        item = {"required_citation": {"ayah": 7}}
        assert _check_citation(item, "الآية ٧ من السورة") is True

    def test_refusal_detected(self):
        assert _check_refusal("لا أعلم ولا يمكن الجزم في هذا") is True

    def test_confident_answer_not_refusal(self):
        assert _check_refusal("الحكم جائز قطعًا بلا خلاف") is False


class TestRunSunniBenchmark:
    def _write_benchmark(self, tmp_path, items):
        path = tmp_path / "bench.jsonl"
        path.write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8"
        )
        return str(path)

    def test_auto_scoring_and_human_queue(self, tmp_path):
        items = [
            {
                "id": "c1",
                "category": "quran_exact_match",
                "question": "Q?",
                "context": "",
                "must_cite": True,
                "must_refuse": False,
                "required_citation": {"surah": 2, "ayah": 255},
                "checker": "citation",
                "reference_answer": "",
            },
            {
                "id": "r1",
                "category": "refusal_calibration",
                "question": "Q?",
                "context": "",
                "must_cite": False,
                "must_refuse": True,
                "required_citation": None,
                "checker": "refusal",
                "reference_answer": "",
            },
            {
                "id": "h1",
                "category": "aqeedah_sunni",
                "question": "Q?",
                "context": "",
                "must_cite": False,
                "must_refuse": False,
                "required_citation": None,
                "checker": "human",
                "reference_answer": "",
            },
        ]
        path = self._write_benchmark(tmp_path, items)

        evaluator = MagicMock()
        evaluator.generate.return_value = (
            "الآية المذكورة هي رقم 255 في سورة البقرة. المصدر: القرآن."
        )
        result = run_sunni_benchmark(evaluator, benchmark_path=path)

        assert result["n_items"] == 3
        assert result["n_auto_scored"] == 2
        assert len(result["human_queue"]) == 1
        assert result["human_queue"][0]["id"] == "h1"
        assert result["categories"]["quran_exact_match"]["accuracy"] == 1.0
        # Confident answer to a must-refuse item must fail.
        assert result["categories"]["refusal_calibration"]["accuracy"] == 0.0
        assert evaluator.generate.call_count == 3

    def test_context_included_in_prompt(self, tmp_path):
        items = [
            {
                "id": "rag1",
                "category": "rag_faithfulness",
                "question": "ماذا قال النص؟",
                "context": "نص تجريبي عن الوقف.",
                "must_cite": False,
                "must_refuse": False,
                "required_citation": None,
                "checker": "human",
                "reference_answer": "",
            }
        ]
        path = self._write_benchmark(tmp_path, items)
        evaluator = MagicMock()
        evaluator.generate.return_value = "x"
        run_sunni_benchmark(evaluator, benchmark_path=path)
        prompt = evaluator.generate.call_args[0][0]
        assert "نص تجريبي" in prompt and "السؤال:" in prompt
