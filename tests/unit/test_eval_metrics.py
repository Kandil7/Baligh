"""Tests for evaluation metrics."""

from baligh.evaluation.metrics import compute_exact_match


class TestComputeExactMatch:
    def test_exact_match(self):
        predictions = ["hello", "world"]
        references = ["hello", "world"]
        assert compute_exact_match(predictions, references) == 1.0

    def test_no_match(self):
        predictions = ["hello", "world"]
        references = ["foo", "bar"]
        assert compute_exact_match(predictions, references) == 0.0

    def test_partial_match(self):
        predictions = ["hello", "world"]
        references = ["hello", "foo"]
        assert compute_exact_match(predictions, references) == 0.5

    def test_empty(self):
        assert compute_exact_match([], []) == 0.0

    def test_case_sensitive(self):
        predictions = ["Hello"]
        references = ["hello"]
        assert compute_exact_match(predictions, references) == 0.0

    def test_whitespace_handling(self):
        predictions = ["  hello  "]
        references = ["hello"]
        assert compute_exact_match(predictions, references) == 1.0
