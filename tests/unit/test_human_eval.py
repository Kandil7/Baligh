"""Tests for human evaluation."""

from baligh.evaluation.human_eval import EvaluationRubric, HumanEvaluation, HumanEvaluator


class TestEvaluationRubric:
    def test_default_rubric(self):
        rubric = EvaluationRubric()
        assert rubric.correctness is not None
        assert rubric.clarity is not None
        assert rubric.arabic_quality is not None
        assert rubric.usefulness is not None
        assert rubric.faithfulness is not None

    def test_custom_rubric(self):
        rubric = EvaluationRubric(correctness="Custom", clarity="Custom")
        assert rubric.correctness == "Custom"
        assert rubric.clarity == "Custom"


class TestHumanEvaluation:
    def test_creation(self):
        eval_obj = HumanEvaluation(
            prompt="test",
            response="test response",
            scores={"correctness": 4, "clarity": 5},
            annotator="user1",
            notes="good",
        )
        assert eval_obj.prompt == "test"
        assert eval_obj.scores["correctness"] == 4
        assert eval_obj.annotator == "user1"
        assert eval_obj.notes == "good"

    def test_default_notes(self):
        eval_obj = HumanEvaluation(
            prompt="test",
            response="r",
            scores={},
            annotator="u",
        )
        assert eval_obj.notes == ""


class TestHumanEvaluator:
    def test_creation(self):
        evaluator = HumanEvaluator()
        assert len(evaluator.evaluations) == 0
        assert isinstance(evaluator.rubric, EvaluationRubric)

    def test_add_evaluation(self):
        evaluator = HumanEvaluator()
        evaluator.add_evaluation(
            prompt="test",
            response="response",
            scores={"correctness": 4},
            annotator="user1",
        )
        assert len(evaluator.evaluations) == 1

    def test_get_average_scores(self):
        evaluator = HumanEvaluator()
        evaluator.add_evaluation(
            prompt="t",
            response="r1",
            scores={"correctness": 4, "clarity": 5},
            annotator="u1",
        )
        evaluator.add_evaluation(
            prompt="t",
            response="r2",
            scores={"correctness": 3, "clarity": 4},
            annotator="u2",
        )
        avg = evaluator.get_average_scores()
        assert avg["correctness"] == 3.5
        assert avg["clarity"] == 4.5

    def test_get_average_empty(self):
        evaluator = HumanEvaluator()
        assert evaluator.get_average_scores() == {}

    def test_export_csv(self, tmp_path):
        evaluator = HumanEvaluator()
        evaluator.add_evaluation(
            prompt="test",
            response="response",
            scores={"correctness": 4},
            annotator="user1",
        )
        csv_path = tmp_path / "eval.csv"
        evaluator.export_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "correctness" in content
        assert "test" in content
