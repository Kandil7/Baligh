#!/usr/bin/env python3
"""Convert the local Islamweb fatwa corpus into Baligh's canonical SFT schema.

The raw corpus at ``D:\\AI\\Datasets\\Custom\\islamweb1`` holds 50 JSON files
keyed by fatwa URL, each value carrying ``{title, question, answer, lang,
related, categories}``. This script projects those rows onto the canonical
``{instruction, input, output}`` schema that ``baligh.data.standardize_sft_dataset``
and the SFT formatter expect, and performs a leakage-safe split.

Why the split happens here and not in ``prepare_data``:

    ``prepare_data._split_dataset`` splits at the ROW level after mixing. The
    same fatwa can appear under more than one language URL (``/ar/fatwa/<id>``
    and ``/en/fatwa/<id>`` are different URLs for one ruling), so a row-level
    split can place one fatwa on both sides of the boundary. Splitting by
    ``fatwa_id`` here keeps every ruling wholly on one side.

Cleaning applied:

* Collapse the whitespace runs the scraper left behind (``\\r\\n`` + tabs).
* Strip the site-chrome label (``الإجابــة`` / ``Answer`` / ``Réponse`` / ...)
  that prefixes every answer. The hamdala opening that follows is authentic
    fatwa style and is KEPT.
* Drop answers shorter than 80 characters -- these are site error pages, not
    rulings (4 rows in this corpus).
* Cap the answer length at a sentence boundary so no completion is cut
    mid-word, which would teach the model to stop mid-sentence.

Usage:
    python -m src.scripts.convert_islamweb
    python -m src.scripts.convert_islamweb --non-arabic-cap 5000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Windows consoles default to cp1252 and crash on Arabic output; the JSONL
# artifacts are always written utf-8, only the console summary needs this.
def _relax_console_encoding() -> None:
    """Let the console emit replacement chars instead of raising UnicodeEncodeError."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:
        return
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    if encoding != "utf8":
        reconfigure(errors="replace")


_relax_console_encoding()

DEFAULT_SOURCE = Path(r"D:\AI\Datasets\Custom\islamweb1")
DEFAULT_OUTPUT = Path(r"D:\AI\datasets\baligh\islamweb_fatwa")

# The task framing is constant across every row: instruction carries the job,
# input carries the question, output carries the ruling.
INSTRUCTION = (
    "أجب عن السؤال الفقهي التالي اعتمادًا على ما يرجحه أهل السنة والجماعة، "
    "واذكر الأدلة حيث تيسر، ووضّح محلّ الخلاف إن وُجد."
)

# Site-chrome labels the scraper prefixed to every answer. Matched after
# whitespace collapse, so the label is always the first token(s).
_CHROME_LABELS = (
    "الإجابــة",
    "الإجابة",
    "الجواب",
    "الجوابــة",
    "Answer",
    "Réponse",
    "Respuesta",
    "Antwort",
    "Jawab",
)
_CHROME_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(x) for x in _CHROME_LABELS) + r")\s*",
    flags=re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_FATWA_ID_RE = re.compile(r"/fatwa/(\d+)")

# Sentence-final punctuation used to truncate long answers without cutting a
# word in half. Arabic uses both ASCII and Arabic-Indic variants.
_SENTENCE_END = tuple(".!؟?…:؛")


def extract_fatwa_id(url: str) -> str | None:
    """Return the numeric fatwa ID from an Islamweb URL, or None."""
    m = _FATWA_ID_RE.search(url)
    return m.group(1) if m else None


def clean_text(text: str | None) -> str:
    """Collapse scraper whitespace runs and strip the ends."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def strip_chrome(answer: str) -> str:
    """Remove the site-chrome label, keeping the hamdala opening intact."""
    return _CHROME_RE.sub("", answer, count=1).strip()


def truncate_at_sentence(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate *text* at *max_chars*, preferring a sentence boundary.

    Returns ``(text, was_truncated)``. Falls back to a hard cut only when no
    sentence boundary exists in the final 20% of the window, so the cut never
    lands mid-word when a natural break is available.
    """
    if len(text) <= max_chars:
        return text, False

    window = text[:max_chars]
    floor = int(max_chars * 0.8)
    for i in range(len(window) - 1, floor, -1):
        if window[i] in _SENTENCE_END:
            return window[: i + 1].strip(), True
    return window.rsplit(" ", 1)[0].strip(), True


def iter_raw_records(source_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load every ``{url: record}`` pair from the 50 source JSON files."""
    files = sorted(source_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files under {source_dir}")

    records: list[tuple[str, dict[str, Any]]] = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name}: expected an object keyed by URL")
        records.extend(payload.items())
    return records


def to_canonical(
    url: str,
    record: dict[str, Any],
    max_answer_chars: int,
    min_answer_chars: int,
) -> dict[str, Any] | None:
    """Project one raw fatwa onto the canonical SFT schema, or None to drop."""
    answer = clean_text(record.get("answer"))
    answer = strip_chrome(answer)
    question = clean_text(record.get("question"))

    if len(answer) < min_answer_chars:
        return None
    if len(question) < 5:
        return None

    answer, _truncated = truncate_at_sentence(answer, max_answer_chars)

    return {
        "instruction": INSTRUCTION,
        "input": question,
        "output": answer,
        "lang": record.get("lang") or "",
        "fatwa_id": extract_fatwa_id(url) or "",
        "url": url,
        "source": "islamweb",
    }


def apply_language_cap(
    rows: list[dict[str, Any]],
    non_arabic_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep every Arabic row, seeded-sample the rest down to *non_arabic_cap*.

    Sampling is proportional to each language's share of the corpus, so the
    cap bounds total size without reweighting English against French.
    """
    arabic = [r for r in rows if r["lang"] == "ar"]
    other = [r for r in rows if r["lang"] != "ar"]

    if non_arabic_cap <= 0 or len(other) <= non_arabic_cap:
        return rows

    rng = random.Random(seed)
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in other:
        by_lang.setdefault(row["lang"], []).append(row)

    picked: list[dict[str, Any]] = []
    for _lang, group in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        share = round(non_arabic_cap * (len(group) / len(other)))
        rng.shuffle(group)
        picked.extend(group[: max(share, 1)])

    # Rounding can overshoot or undershoot the cap; trim deterministically.
    rng.shuffle(picked)
    picked = picked[:non_arabic_cap]
    return arabic + picked


def dedupe_by_fatwa_id(
    rows: list[dict[str, Any]],
    language_priority: tuple[str, ...] = ("ar", "en", "fr", "de", "es", "id"),
) -> list[dict[str, Any]]:
    """Keep one row per fatwa ID, preferring Arabic.

    The same ruling can appear under several language URLs (``/ar/fatwa/<id>``
    and ``/en/fatwa/<id>``). Leaving both in the corpus lets a later ROW-level
    split -- ``prepare_data._split_dataset`` splits the mixed set by row --
    put one ruling on both sides of the boundary. Collapsing to a single row
    per ID makes every subsequent split safe, whatever splits it.
    """
    rank = {lang: i for i, lang in enumerate(language_priority)}
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("fatwa_id")
        if not key:
            # No parseable ID: fall back to the URL so these still dedupe.
            key = row.get("url") or ""
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if rank.get(row.get("lang", ""), len(rank)) < rank.get(current.get("lang", ""), len(rank)):
            best[key] = row
    return list(best.values())


def group_split(
    rows: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by ``fatwa_id`` so one ruling never straddles train and eval.

    Rows without a parseable fatwa ID fall back to their URL as the group key,
    which still keeps each distinct ruling on one side.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("fatwa_id") or row.get("url") or ""
        groups.setdefault(key, []).append(row)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    target_eval = len(rows) * eval_ratio
    running = 0
    for key in keys:
        group = groups[key]
        if running < target_eval:
            eval_rows.extend(group)
            running += len(group)
        else:
            train_rows.extend(group)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    source_dir: Path,
    output_dir: Path,
    seed: int,
    non_arabic_cap: int,
    eval_ratio: float,
    min_answer_chars: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    """Run the full conversion and write train/eval JSONL plus a manifest."""
    raw = iter_raw_records(source_dir)

    rows: list[dict[str, Any]] = []
    dropped_short = 0
    for url, record in raw:
        row = to_canonical(url, record, max_answer_chars, min_answer_chars)
        if row is None:
            dropped_short += 1
            continue
        rows.append(row)

    before_cap = len(rows)
    rows = apply_language_cap(rows, non_arabic_cap, seed)

    before_dedupe = len(rows)
    rows = dedupe_by_fatwa_id(rows)

    train_rows, eval_rows = group_split(rows, eval_ratio, seed)

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)

    fatwa_ids = [r["fatwa_id"] for r in rows if r["fatwa_id"]]
    repeated_ids = {k: v for k, v in Counter(fatwa_ids).items() if v > 1}

    train_groups = {r["fatwa_id"] or r["url"] for r in train_rows}
    eval_groups = {r["fatwa_id"] or r["url"] for r in eval_rows}
    overlap = train_groups & eval_groups

    lang_counts = Counter(r["lang"] for r in rows)
    manifest = {
        "source": str(source_dir),
        "output": str(output_dir),
        "seed": seed,
        "raw_rows": len(raw),
        "dropped_short": dropped_short,
        "rows_before_cap": before_cap,
        "rows_after_cap": len(rows),
        "rows_before_dedupe": before_dedupe,
        "rows_after_dedupe": len(rows),
        "deduped_away": before_dedupe - len(rows),
        "non_arabic_cap": non_arabic_cap,
        "language_counts": dict(lang_counts.most_common()),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "group_overlap": len(overlap),
        "repeated_fatwa_ids": len(repeated_ids),
        "min_answer_chars": min_answer_chars,
        "max_answer_chars": max_answer_chars,
        # Every fatwa ID now appears exactly once, so a later ROW-level split
        # (prepare_data._split_dataset) cannot straddle a ruling either.
        "row_level_split_safe": len(repeated_ids) == 0,
        "leakage_check": "PASS" if not overlap else "FAIL",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the local Islamweb fatwa corpus to canonical SFT JSONL"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--non-arabic-cap",
        type=int,
        default=3000,
        help="Max non-Arabic rows kept (Arabic is always kept in full)",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=80,
        help="Answers below this are site error pages, not rulings",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=8000,
        help="Truncate longer answers at a sentence boundary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        source_dir=args.source,
        output_dir=args.output,
        seed=args.seed,
        non_arabic_cap=args.non_arabic_cap,
        eval_ratio=args.eval_ratio,
        min_answer_chars=args.min_answer_chars,
        max_answer_chars=args.max_answer_chars,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if manifest["leakage_check"] != "PASS":
        print("LEAKAGE CHECK FAILED: fatwa groups straddle train and eval", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
