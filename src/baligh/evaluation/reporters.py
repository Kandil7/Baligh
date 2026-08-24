"""Report generation for Baligh-1.7B v0."""

import json
import re
from datetime import datetime
from pathlib import Path

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_name(name: str) -> str:
    """Make a model/repo name filename-safe (HF repo ids contain '/')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def generate_eval_report(
    results: dict,
    output_dir: str | Path,
    model_name: str = "baligh-1.7b-v0",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"model": model_name, "timestamp": datetime.now().isoformat(), "results": results}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{_safe_name(model_name)}_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Eval report saved to {report_path}")
    return report_path


def save_results(results: dict, output_dir: str | Path, prefix: str = "eval") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"{prefix}_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {results_path}")
    return results_path
