"""Human evaluation for Baligh-1.7B v0."""

from dataclasses import asdict, dataclass
from pathlib import Path

from baligh.config import get_eval_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

MIN_SCORE = 1
MAX_SCORE = 5


@dataclass
class EvaluationRubric:
    correctness: str = "1-5: Factual accuracy of the response"
    clarity: str = "1-5: Clarity and readability of Arabic"
    arabic_quality: str = "1-5: Formal Arabic quality (fusha)"
    usefulness: str = "1-5: Helpfulness for the user query"
    faithfulness: str = "1-5: Faithfulness to source knowledge"

    @classmethod
    def default(cls) -> "EvaluationRubric":
        config = get_eval_config()
        return cls(**config.human_eval_rubric)

    @property
    def criteria(self) -> list[str]:
        """Rubric keys in declaration order - the single source of truth
        for CSV columns and averages."""
        return list(asdict(self).keys())


@dataclass
class HumanEvaluation:
    prompt: str
    response: str
    scores: dict[str, int]
    annotator: str
    notes: str = ""


class HumanEvaluator:
    def __init__(self, rubric: "EvaluationRubric | None" = None) -> None:
        self.rubric = rubric or EvaluationRubric.default()
        self.evaluations: list[HumanEvaluation] = []

    def add_evaluation(
        self,
        prompt: str,
        response: str,
        scores: dict[str, int],
        annotator: str,
        notes: str = "",
    ) -> None:
        """Add an evaluation; scores are validated against the rubric."""
        for key, value in scores.items():
            if key not in self.rubric.criteria:
                raise ValueError(f"Unknown rubric key: {key}")
            if not MIN_SCORE <= int(value) <= MAX_SCORE:
                raise ValueError(f"Score for {key!r} must be {MIN_SCORE}-{MAX_SCORE}, got {value}")
        eval_obj = HumanEvaluation(
            prompt=prompt,
            response=response,
            scores={k: int(v) for k, v in scores.items()},
            annotator=annotator,
            notes=notes,
        )
        self.evaluations.append(eval_obj)

    def get_average_scores(self) -> dict[str, float]:
        """Average per rubric criterion over evaluations that supplied it.

        Keyed by the RUBRIC (not the first evaluation's dict), so annotators
        submitting different key subsets can never KeyError or misalign.
        """
        if not self.evaluations:
            return {}
        averages = {}
        for criterion in self.rubric.criteria:
            values = [e.scores[criterion] for e in self.evaluations if criterion in e.scores]
            if values:
                averages[criterion] = sum(values) / len(values)
        return averages

    def export_csv(self, path: str | Path) -> None:
        """Export with columns fixed by the rubric; missing scores stay blank
        so rows can never misalign with the header."""
        import csv

        criteria = self.rubric.criteria
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "response"] + criteria + ["annotator", "notes"])
            for e in self.evaluations:
                writer.writerow(
                    [e.prompt, e.response]
                    + [e.scores.get(c, "") for c in criteria]
                    + [e.annotator, e.notes]
                )
        logger.info(f"Human eval exported to {path}")
