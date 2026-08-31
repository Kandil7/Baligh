#!/usr/bin/env python3
"""Prepare Baligh-1.7B SFT data for Qwen3-1.7B.

Downloads public HF datasets, normalizes schemas, deduplicates Arabic text,
keeps provenance/license metadata, performs source-level splitting, and writes
ready JSONL plus a review queue. It never fabricates answers from raw passages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

# Windows consoles default to cp1252 and crash on Arabic output; the JSONL
# artifacts are always written utf-8, only the console summary needs this.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(errors="replace")

MODEL = "Qwen/Qwen3-1.7B"
SYSTEM = (
    "أنت بليغ، مساعد عربي علمي ملتزم بمنهج أهل السنة والجماعة. "
    "أجب اعتمادًا على المصدر والسياق، وميّز بين النص المنقول والشرح والاستنباط. "
    "لا تنسب آية أو حديثًا أو قولًا إلى الله أو رسوله أو عالم دون توثيق. "
    "عند وجود خلاف معتبر اذكره، ولا تعرض قولًا واحدًا على أنه إجماع. "
    "إذا لم يكف السياق أو لم تثبت المعلومة فقل: لا أعلم أو لا يكفي السياق للجزم. "
    "في الفتوى الشخصية نبّه إلى مراجعة عالم موثوق."
)

# Per-source policy matrix. `policy` is the deliberate train-vs-review
# decision (review finding C1) — never inferred inline:
#   "ready"  → enters train.jsonl (non-religious or already reviewed).
#   "review" → enters needs_review.jsonl until a human Sunni review passes.
# `--promote <source_id>` moves a review source into train while recording
# the promotion in the manifest, so the decision is always explicit.
SOURCES: list[dict[str, Any]] = [
    {
        "id": "Raniahossam33/Islamweb_part2",
        "mode": "fatwa",
        "split": "train",
        "weight": 1.0,
        "license_note": "verify source terms",
        "policy": "review",  # fatwas: mandatory Sunni review
    },
    {
        "id": "NightPrince/islamic-arabic-qa",
        "mode": "chat",
        "split": "train",
        "weight": 1.0,
        "license_note": "Apache-2.0 metadata",
        "policy": "ready",  # instruction-formatted Islamic QA; spot-audit before release
    },
    {
        "id": "Omar-youssef/islamic-qa-egyptian-arabic",
        "mode": "chat",
        "split": "train",
        "weight": 0.30,
        "license_note": "Apache-2.0 metadata",
        "policy": "ready",  # dialect comprehension; capped small by weight
    },
    {
        "id": "bobez999/arabic-qa-dataset-sigir2024",
        "mode": "instruction",
        "split": "train",
        "weight": 0.25,
        "license_note": "MIT metadata",
        "policy": "ready",  # general Arabic QA, no religious claims
    },
    {
        "id": "Kandil7/Athar-RAG-Hub",
        "mode": "rag_review",
        "split": "train",
        "weight": 0.50,
        "license_note": "CC-BY-NC-4.0; review commercial use",
        "policy": "review",  # raw passages: answers must be authored + reviewed
    },
]

Q_KEYS = ("question", "Question", "instruction", "prompt", "query", "user", "input")
A_KEYS = ("answer", "Answer", "response", "output", "assistant", "completion")
C_KEYS = ("context", "Question_Context", "passage", "source_text", "content")
S_KEYS = ("source", "URL", "url", "citation", "book_title", "title")


def s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(s(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    value = str(v).strip()
    return "" if value.lower() in {"nan", "none", "null"} else value


def pick(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row and s(row[k]):
            return s(row[k])
    lowered = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        if k.lower() in lowered and s(lowered[k.lower()]):
            return s(lowered[k.lower()])
    return ""


def norm(v: str) -> str:
    v = unicodedata.normalize("NFKC", v)
    v = re.sub(r"[\u064B-\u065F\u0670]", "", v)
    v = v.replace("ـ", "")
    v = re.sub(r"[إأآٱ]", "ا", v).replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    v = re.sub(r"https?://\S+|www\.\S+", " URL ", v)
    return re.sub(r"\s+", " ", v).strip()


def digest(*parts: str) -> str:
    return hashlib.sha256("\n".join(norm(x) for x in parts).encode()).hexdigest()


def make(
    user: str,
    answer: str,
    source: str,
    kind: str,
    citation: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one example. `needs_sunni_review` is stamped centrally in
    build() from the source's policy — row functions never decide routing."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "source_dataset": source,
        "source_url": citation if citation.startswith(("http://", "https://")) else "",
        "citation": citation,
        "example_type": kind,
        "needs_sunni_review": False,  # overwritten by source policy
        "metadata": meta or {},
        "dedup_key": digest(user, answer),
    }


def chat_row(row: dict[str, Any], repo: str, kind: str) -> dict[str, Any] | None:
    q, a = pick(row, Q_KEYS), pick(row, A_KEYS)
    if not q or not a or len(norm(a)) < 20:
        return None
    context = pick(row, C_KEYS)
    citation = pick(row, S_KEYS)
    user = q if not context else f"السؤال:\n{q}\n\nالسياق:\n{context[:12000]}"
    if citation and "المصدر" not in a:
        a = f"{a}\n\nالمصدر: {citation}"
    return make(user, a, repo, kind, citation)


def fatwa_row(row: dict[str, Any], repo: str) -> dict[str, Any] | None:
    q, a = pick(row, ("Question",)), pick(row, ("Answer",))
    if not q or not a or len(norm(a)) < 30:
        return None
    context = pick(row, ("Question_Context",))
    url = pick(row, ("URL",))
    number = pick(row, ("Question Number",))
    user = f"السؤال:\n{q}"
    if context:
        user += f"\n\nتفاصيل السائل:\n{context[:12000]}"
    source = f"إسلام ويب، فتوى رقم {number}" if number else "إسلام ويب"
    answer = f"{a}\n\nالمصدر: {source}"
    if url:
        answer += f"\nالرابط: {url}"
    return make(user, answer, repo, "fatwa_qa", url, meta={"question_number": number})


def passage_review(row: dict[str, Any], repo: str) -> dict[str, Any] | None:
    passage = pick(row, ("text", "content", "passage"))
    if len(norm(passage)) < 150:
        return None
    title = pick(row, ("book_title", "title"))
    author = pick(row, ("author",))
    page = pick(row, ("page_number", "page", "page_range"))
    citation = "، ".join(x for x in (title, author, f"ص {page}" if page else "") if x)
    user = (
        "استخرج من النص الآتي خلاصة أمينة، ولا تضف حكمًا أو استنباطًا غير موجود في النص.\n\nالنص:\n"
        + passage[:12000]
    )
    answer = "[تحتاج هذه العينة إلى إجابة يراجعها عالم أو باحث سني قبل إدخالها في تدريب SFT.]"
    return make(user, answer, repo, "passage_review", citation)


def load_repo(
    repo: str, mode: str, split: str, token: str | None, limit: int
) -> tuple[list[dict[str, Any]], bool]:
    try:
        ds = load_dataset(repo, split=split, token=token)
    except Exception as exc:
        print(f"[WARN] {repo}: {exc}")
        return [], True
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for row in ds:
        item = (
            fatwa_row(row, repo)
            if mode == "fatwa"
            else passage_review(row, repo)
            if mode == "rag_review"
            else chat_row(row, repo, mode)
        )
        if item:
            out.append(item)
    return out, False


def cap_and_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def leakage_group_key(row: dict[str, Any]) -> str:
    """Composite validation-group key (review finding C2).

    Splitting only by source_dataset lets same-mas'ala paraphrases land in
    both train and validation. Group by the finest provenance unit available
    (fatwa number / URL / book citation), falling back to the source itself.
    """
    source = row["source_dataset"]
    number = row.get("metadata", {}).get("question_number") or ""
    if number:
        return f"{source}::{number}"
    url = row.get("source_url") or ""
    if url:
        return f"{source}::url:{url}"
    citation = row.get("citation") or ""
    if citation:
        return f"{source}::cite:{citation}"
    return source


def build(args: argparse.Namespace) -> None:
    token = os.getenv("HF_TOKEN")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    promote = set(args.promote or [])
    unknown_promote = promote - {src["id"] for src in SOURCES}
    if unknown_promote:
        raise SystemExit(f"--promote: unknown source IDs: {sorted(unknown_promote)}")

    rows: list[dict[str, Any]] = []
    sources_failed: list[str] = []
    for src in SOURCES:
        repo, mode = src["id"], src["mode"]
        loaded, failed = load_repo(repo, mode, src["split"], token, args.limit_per_source)
        if failed:
            sources_failed.append(repo)  # recorded, not swallowed (finding C3)
        weight = float(src["weight"])
        if 0 < weight < 1 and loaded:
            loaded = cap_and_sample(loaded, max(1, int(len(loaded) * weight)), args.seed)
        policy = "ready" if repo in promote else src["policy"]
        for item in loaded:
            item["license_note"] = src["license_note"]
            item["needs_sunni_review"] = policy == "review"
            item["source_policy"] = policy
        rows.extend(loaded)
        print(f"{repo}: {len(loaded)} ({policy})")

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        if row["dedup_key"] not in seen:
            seen.add(row["dedup_key"])
            unique.append(row)
    rng = random.Random(args.seed)
    rng.shuffle(unique)

    ready = [x for x in unique if not x["needs_sunni_review"]]
    review = [x for x in unique if x["needs_sunni_review"]]
    # Composite-group split: one mas'ala/fatwa/book never straddles
    # train and validation.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ready:
        groups.setdefault(leakage_group_key(row), []).append(row)
    train, val = [], []
    for group in groups.values():
        # A singleton group must stay in train — max(1, ...) would otherwise
        # let tiny groups vanish entirely into validation.
        n = (
            min(len(group) - 1, max(1, round(len(group) * args.validation_ratio)))
            if len(group) > 1
            else 0
        )
        val.extend(group[:n])
        train.extend(group[n:])

    Dataset.from_list(train).to_json(str(root / "train.jsonl"), force_ascii=False)
    Dataset.from_list(val).to_json(str(root / "validation.jsonl"), force_ascii=False)
    Dataset.from_list(review).to_json(str(root / "needs_review.jsonl"), force_ascii=False)
    manifest = {
        "base_model": MODEL,
        "system_prompt": SYSTEM,
        "sources": [x["id"] for x in SOURCES],
        "sources_failed": sources_failed,
        "promoted_to_train": sorted(promote),
        "rows_total": len(unique),
        "rows_ready": len(ready),
        "rows_needs_review": len(review),
        "rows_train": len(train),
        "rows_validation": len(val),
        "validation_groups": len(groups),
        "source_counts": dict(Counter(x["source_dataset"] for x in unique)),
        "warning": "Review Sunni doctrine, hadith attribution, fiqh claims, citations, and licenses before final training or redistribution.",
    }
    if sources_failed:
        manifest["warning"] += f" FAILED SOURCES (absent from output): {sources_failed}."
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/baligh_final_qwen3_1.7b")
    p.add_argument("--limit-per-source", type=int, default=0, help="0 means all rows")
    p.add_argument("--validation-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--promote",
        action="append",
        metavar="SOURCE_ID",
        help="Move a 'review' source into train despite its policy; "
        "recorded in manifest.promoted_to_train. Repeatable.",
    )
    args = p.parse_args()
    build(args)
