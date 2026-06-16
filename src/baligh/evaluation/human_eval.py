"""Human evaluation for Baligh-1.5B v0."""

from dataclasses import dataclass, field
from typing import List, Dict
from baligh.config import get_eval_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class EvaluationRubric:
    correctness: str = "1-5: Factual accuracy of the response"
    clarity: str = "1-5: Clarity and readability of Arabic"
    arabic_quality: str = "1-5: Formal Arabic quality (fusha)"
    usefulness: str = "1-5: Helpfulness for the user query"
    faithfulness: str = "1-5: Faithfulness to source knowledge"
    
    @classmethod
    def default(cls):
        config = get_eval_config()
        return cls(**config.human_eval_rubric)

@dataclass
class HumanEvaluation:
    prompt: str
    response: str
    scores: Dict[str, int]
    annotator: str
    notes: str = ""

class HumanEvaluator:
    def __init__(self, rubric=None):
        self.rubric = rubric or EvaluationRubric.default()
        self.evaluations: List[HumanEvaluation] = []

    def add_evaluation(self, prompt, response, scores, annotator, notes=""):
        eval_obj = HumanEvaluation(prompt=prompt, response=response, scores=scores, annotator=annotator, notes=notes)
        self.evaluations.append(eval_obj)

    def get_average_scores(self):
        if not self.evaluations:
            return {}
        keys = self.evaluations[0].scores.keys()
        return {k: sum(e.scores[k] for e in self.evaluations) / len(self.evaluations) for k in keys}

    def export_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "response"] + list(self.rubric.__dict__.keys()) + ["annotator", "notes"])
            for e in self.evaluations:
                writer.writerow([e.prompt, e.response] + list(e.scores.values()) + [e.annotator, e.notes])
        logger.info("Human eval exported to %s" % path)
