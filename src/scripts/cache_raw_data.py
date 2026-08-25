#!/usr/bin/env python3
"""Cache raw datasets from the Baligh registry as local JSONL files.

RAW LAYER ONLY: no cleaning, no normalization — tashkeel is preserved and
original column values are kept verbatim. Purpose: offline reruns without
re-downloading web-scale corpora on every pipeline execution.

Column projection is registry-driven (never hardcoded):
- CPT / cpt_islamic sources -> ``text`` (plus question/answer when the
  source defines them, e.g. nazimali/quran-question-answer-context).
- SFT sources -> the registry's instruction/input/output columns verbatim
  (e.g. summarization keeps its native text/summary mapping untouched).

Each output line: original payload fields + metadata
(source, hf_path, license, domain, type).
"""

import argparse
import json
from pathlib import Path

from baligh.data.datasets import DATASETS, get_dataset_config
from baligh.data.loader import load_dataset_by_name
from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_OUTPUT_ROOT = "D:/AI/Datasets/baligh"

STAGE_SUBDIRS = {
    "cpt": "cpt/general",
    "cpt_islamic": "cpt/islamic",
    "sft": "sft/raw",
}

OPTIONAL_TEXT_EXTRAS = ("question", "answer")


def _payload_columns(cfg: dict) -> dict[str, str]:
    """Map output field -> source column, straight from the registry entry."""
    if "instruction_column" in cfg:
        fields = {"instruction": cfg["instruction_column"]}
        if cfg.get("input_column"):
            fields["input"] = cfg["input_column"]
        fields["output"] = cfg["output_column"]
        return fields
    fields = {"text": cfg.get("text_column", "text")}
    return fields


def _metadata(cfg: dict) -> dict:
    hf_path = cfg["path"]
    if cfg.get("name"):
        hf_path = f"{hf_path}:{cfg['name']}"
    return {
        "hf_path": hf_path,
        "license": cfg.get("license", "unknown"),
        "domain": cfg.get("domain", "unknown"),
        "type": cfg.get("type", "unknown"),
    }


def _bounded_view(ds, streaming: bool, max_samples: int | None):
    if max_samples is None:
        return ds
    return ds.take(max_samples) if streaming else ds.select(range(min(max_samples, len(ds))))


def cache_dataset(
    name: str, out_path: Path, max_samples: int | None, overwrite: bool
) -> dict | None:
    cfg = get_dataset_config(name)
    if out_path.exists() and not overwrite:
        logger.info(f"Skip (exists): {out_path}")
        return None

    ds = load_dataset_by_name(
        name, split=cfg.get("split", "train"), streaming=cfg.get("streaming", False)
    )
    ds = _bounded_view(ds, cfg.get("streaming", False), max_samples)

    fields = _payload_columns(cfg)
    extras = [c for c in OPTIONAL_TEXT_EXTRAS if c in (ds.column_names or [])]
    meta = {"source": name, **_metadata(cfg)}

    rows = 0
    chars = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = {field: row.get(col) or "" for field, col in fields.items()}
            for col in extras:
                record[col] = row.get(col) or ""
            record.update(meta)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
            chars += sum(len(str(v)) for v in record.values())

    stats = {"file": str(out_path), "rows": rows, "chars": chars}
    logger.info(f"Cached {name}: {rows} rows -> {out_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Cache raw Baligh datasets as JSONL")
    parser.add_argument(
        "--stage",
        choices=["cpt", "sft", "both"],
        default="both",
        help="Which dataset families to cache",
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-samples", type=int, default=None, help="Cap rows per dataset")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()
    setup_logging()

    stages = ["cpt", "cpt_islamic", "sft"] if args.stage == "both" else [args.stage]
    manifests: dict[str, dict] = {}

    for stage in stages:
        subdir = STAGE_SUBDIRS[stage]
        names = sorted(n for n, c in DATASETS.items() if c.get("type") == stage)
        entries = {}
        for name in names:
            try:
                stats = cache_dataset(
                    name,
                    args.output_root / subdir / f"{name}.jsonl",
                    args.max_samples,
                    args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - one bad repo must not kill the batch
                logger.error(f"Failed to cache {name}: {exc}")
                continue
            if stats:
                entries[name] = stats
        if entries:
            manifest_path = args.output_root / subdir / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            manifests[subdir] = entries
            logger.info(f"Manifest: {manifest_path}")

    total_rows = sum(e["rows"] for m in manifests.values() for e in m.values())
    logger.info(f"Done. {total_rows} rows cached across {len(manifests)} stage dirs.")


if __name__ == "__main__":
    main()
