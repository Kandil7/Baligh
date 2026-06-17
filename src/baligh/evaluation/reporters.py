"""Report generation for Baligh-1.5B v0."""

import json
from datetime import datetime
from pathlib import Path

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def generate_eval_report(results, output_dir, model_name="baligh-1.5b-v0"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"model": model_name, "timestamp": datetime.now().isoformat(), "results": results}

    report_path = (
        output_dir / f"eval_report_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Eval report saved to %s" % report_path)
    return report_path


def save_results(results, output_dir, prefix="eval"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{prefix}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Results saved to %s" % results_path)
    return results_path
