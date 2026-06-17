"""Tests for evaluation reporters."""

import json
from pathlib import Path

from baligh.evaluation.reporters import generate_eval_report, save_results


class TestGenerateEvalReport:
    def test_creates_report(self, tmp_path):
        results = {"mmlu": {"accuracy": 0.65}, "cidar": {"rouge_l": 0.42}}
        report_path = generate_eval_report(results, tmp_path, model_name="test-model")
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)
        assert report["model"] == "test-model"
        assert "mmlu" in report["results"]
        assert "timestamp" in report


class TestSaveResults:
    def test_saves_results(self, tmp_path):
        results = {"accuracy": 0.65, "loss": 2.3}
        path = save_results(results, tmp_path, prefix="test")
        assert path.exists()
        assert "test" in path.name

        with open(path) as f:
            data = json.load(f)
        assert data["accuracy"] == 0.65

    def test_saves_with_prefix(self, tmp_path):
        results = {"accuracy": 0.65}
        path = save_results(results, tmp_path, prefix="mmlu")
        assert "mmlu" in path.name
