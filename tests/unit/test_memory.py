"""Tests for memory utilities."""

import pytest
import torch

from baligh.utils.memory import (
    clear_memory,
    estimate_training_memory,
    log_memory_stats,
    print_model_memory,
)


class TestClearMemory:
    def test_does_not_crash(self):
        clear_memory()


class TestLogMemoryStats:
    def test_returns_dict(self):
        result = log_memory_stats(prefix="Test")
        assert isinstance(result, dict)


class TestEstimateTrainingMemory:
    def test_returns_dict(self):
        result = estimate_training_memory(
            model_params=1500000000,
            batch_size=2,
            seq_length=2048,
        )
        assert isinstance(result, dict)
        assert "total_gb" in result
        assert "model_gb" in result
        assert "gradients_gb" in result
        assert "optimizer_gb" in result
        assert "activations_gb" in result

    def test_with_gradient_checkpointing(self):
        r1 = estimate_training_memory(1500000000, 2, 2048, gradient_checkpointing=False)
        r2 = estimate_training_memory(1500000000, 2, 2048, gradient_checkpointing=True)
        assert r2["activations_gb"] < r1["activations_gb"]


class TestPrintModelMemory:
    def test_does_not_crash(self):
        model = torch.nn.Linear(10, 10)
        print_model_memory(model)
