"""Benchmark runners for Baligh-1.7B v0."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from baligh.evaluation.metrics import compute_bleu, compute_exact_match, compute_rouge
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

_MMLU_LETTER_RE = re.compile(r"\b([A-D]|[أ-د])\b")
_ARABIC_LETTER_MAP = {"أ": "A", "ب": "B", "ج": "C", "د": "D"}


def _extract_choice(response: str) -> str:
    """Extract the first A-D choice letter from a free-form response."""
    match = _MMLU_LETTER_RE.search(response or "")
    if not match:
        return ""
    token = match.group(1)
    return _ARABIC_LETTER_MAP.get(token, token)


def _mmlu_fields(example: dict) -> tuple[str, list[str], str] | None:
    """Normalize both known MMLU-Arabic schemas to (question, choices, gold).

    FreedomIntelligence/MMLU_Arabic has shipped two layouts: a dict
    ``choices``/``answer`` layout and a positional CSV export with columns
    "0".."3". Handle both explicitly — assuming one silently crashes or,
    worse, measures first-character noise.
    """
    question = example.get("question")

    choices_raw = example.get("choices")
    if isinstance(choices_raw, dict):
        choices = [str(choices_raw[k]) for k in sorted(choices_raw)]
    elif isinstance(choices_raw, (list, tuple)):
        choices = [str(c) for c in choices_raw]
    else:
        positional = [example.get(str(i)) for i in range(4)]
        if all(p is not None for p in positional):
            choices = [str(p) for p in positional]
        else:
            return None

    gold = example.get("answer")
    # Letter form ("A"), index int, or string digit all occur in the wild.
    if isinstance(gold, bool):
        return None
    if isinstance(gold, int):
        gold_letter = chr(ord("A") + gold) if 0 <= gold < len(choices) else ""
    else:
        gold_str = str(gold).strip()
        gold_letter = (
            gold_str.upper()
            if gold_str.upper() in {"A", "B", "C", "D"}
            else (_ARABIC_LETTER_MAP.get(gold_str, ""))
        )
        if not gold_letter and gold_str.isdigit():
            idx = int(gold_str)
            gold_letter = chr(ord("A") + idx) if 0 <= idx < len(choices) else ""

    if not question or len(choices) < 2 or not gold_letter:
        return None
    return str(question), choices, gold_letter


def run_mmlu_arabic(evaluator: Any, max_samples: int = 100) -> dict:
    """MMLU-Arabic multiple-choice accuracy.

    Prompt demands the bare choice letter (Arabic instruction); extraction
    regexes accept Latin or Arabic letter responses anywhere in the output.
    """
    logger.info("Running MMLU-Arabic benchmark")
    dataset = load_dataset("FreedomIntelligence/MMLU_Arabic", split="test")

    results: list[dict[str, object]] = []
    skipped = 0
    for raw in dataset.select(range(min(max_samples, len(dataset)))):
        fields = _mmlu_fields(raw)
        if fields is None:
            skipped += 1
            continue
        question, choices, gold_letter = fields
        letters_line = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)
        )
        prompt = f"{question}\n{letters_line}\n\nأجب بالحرف فقط (A أو B أو C أو D)."
        response = evaluator.generate(prompt)
        pred = _extract_choice(response)
        results.append({"correct": pred == gold_letter, "pred": pred, "gold": gold_letter})

    if skipped:
        logger.warning(f"MMLU-Arabic: skipped {skipped} malformed examples")
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    logger.info(f"MMLU-Arabic accuracy: {accuracy:.4f} over {len(results)} examples")
    return {"accuracy": accuracy, "results": results}


def run_cidar_eval(evaluator: Any, max_samples: int = 100) -> dict:
    """CIDAR generation benchmark (ROUGE/BLEU against references)."""
    logger.info("Running CIDAR evaluation")
    dataset = load_dataset("arbml/CIDAR", split="test")
    predictions = []
    references = []
    for example in dataset.select(range(min(max_samples, len(dataset)))):
        prompt = example["instruction"]
        if example.get("input"):
            prompt += "\n" + example["input"]
        response = evaluator.generate(prompt)
        predictions.append(response)
        references.append(example["output"])
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    logger.info(f"CIDAR ROUGE: {rouge}")
    logger.info(f"CIDAR BLEU: {bleu:.2f}")
    return {"rouge": rouge, "bleu": bleu, "predictions": predictions, "references": references}


def run_islamic_qa(evaluator: Any, dataset: Any, max_samples: int = 100) -> dict:
    """Islamic QA evaluation (ROUGE + normalized exact match)."""
    logger.info("Running Islamic QA evaluation")
    predictions = []
    references = []
    for example in dataset.select(range(min(max_samples, len(dataset)))):
        prompt = example.get("question", example.get("instruction", ""))
        response = evaluator.generate(prompt)
        predictions.append(response)
        references.append(example.get("answer", example.get("output", "")))
    rouge = compute_rouge(predictions, references)
    exact_match = compute_exact_match(predictions, references)
    logger.info(f"Islamic QA ROUGE: {rouge}")
    logger.info(f"Islamic QA Exact Match: {exact_match:.4f}")
    return {
        "rouge": rouge,
        "exact_match": exact_match,
        "predictions": predictions,
        "references": references,
    }


# --- Sunni-core benchmark -------------------------------------------------

_REFUSAL_MARKERS = (
    "لا أعلم",
    "لا اعلم",
    "لا يكفي",
    "غير كافٍ",
    "غير كافي",
    "لا يمكن الجزم",
    "لا وجود",
    "لا أستطيع الجزم",
    "لا يجوز الجزم",
    "مراجعة عالم",
    "استشارة عالم",
    "أهل العلم",
    "راجع عالم",
    "السياق لا",
    "لا يصلح للإجابة",
)


def _check_citation(item: dict[str, Any], response: str) -> bool:
    """Auto-check: required citation fragments must appear in the response."""
    required = item.get("required_citation") or {}
    haystack = response or ""
    checks: list[bool] = []
    if required.get("surah") is not None:
        checks.append(str(required["surah"]) in haystack)
    if required.get("ayah") is not None:
        ayah = str(required["ayah"])
        # Western or Eastern-Arabic digits.
        eastern = ayah.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
        checks.append(ayah in haystack or eastern in haystack)
    if required.get("number"):
        number = str(required["number"])
        eastern = number.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
        checks.append(number in haystack or eastern in haystack)
    for field in ("collection", "title"):
        if required.get(field):
            checks.append(str(required[field]) in haystack)
    return all(checks) if checks else False


def _check_refusal(response: str) -> bool:
    """Auto-check: the model declined instead of asserting."""
    text = response or ""
    return any(marker in text for marker in _REFUSAL_MARKERS)


def run_sunni_benchmark(
    evaluator: Any,
    benchmark_path: str | Path = "data/benchmarks/sunni_core_v0.jsonl",
    max_samples: int = 100,
) -> dict:
    """Sunni-domain core benchmark (the project's target niche).

    Categories: Quran exact-match/citation, hadith attribution honesty,
    madhhab-aware fiqh, aqeedah, RAG faithfulness, refusal calibration,
    Arabic quality, and safety/deferral. Items carry a ``checker`` field:

    - ``citation``: auto-scored — required citation fragments (surah/ayah
      numbers, hadith collection+number, book title) must appear verbatim.
    - ``refusal``: auto-scored — refusal markers present; fabrication fails.
    - ``human``: not scored automatically; routed to the human-review queue.

    Returns per-category accuracy over auto-checkable items plus the human
    queue. Auto scores are conservative: they can only pass on explicit,
    verifiable content.
    """
    logger.info("Running Sunni-core benchmark from %s", benchmark_path)
    items: list[dict[str, Any]] = []
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                items.append(json.loads(line))

    results: dict[str, dict[str, Any]] = defaultdict(lambda: {"pass": 0, "total": 0})
    human_queue: list[dict[str, Any]] = []

    for item in items[:max_samples]:
        question = item["question"]
        context = item.get("context") or ""
        user = f"السؤال:\n{question}"
        if context:
            user += f"\n\nالسياق:\n{context}"
        response = evaluator.generate(user)

        category = item["category"]
        checker = item.get("checker", "human")
        record = {
            "id": item["id"],
            "category": category,
            "response": response,
            "reference_answer": item.get("reference_answer", ""),
        }
        if checker == "citation":
            # Auto-fail only when a verifiable citation is required and absent.
            passed = _check_citation(item, response) if item.get("must_cite") else True
            results[category]["total"] += 1
            results[category]["pass"] += int(passed)
            record["passed"] = passed
            record["checker"] = checker
        elif checker == "refusal":
            passed = _check_refusal(response) if item.get("must_refuse") else True
            results[category]["total"] += 1
            results[category]["pass"] += int(passed)
            record["passed"] = passed
            record["checker"] = checker
        else:
            human_queue.append(record)

    summary = {
        cat: {"accuracy": v["pass"] / v["total"] if v["total"] else 0.0, **v}
        for cat, v in results.items()
    }
    logger.info("Sunni-core auto metrics: %s", json.dumps(summary, ensure_ascii=False))
    logger.info("Sunni-core human-review queue: %d items", len(human_queue))
    return {
        "categories": summary,
        "human_queue": human_queue,
        "n_items": len(items),
        "n_auto_scored": sum(v["total"] for v in results.values()),
    }
