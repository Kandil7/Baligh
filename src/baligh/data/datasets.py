"""Dataset registry for Baligh-1.7B v0.

This module is the SINGLE source of truth for dataset metadata (paths,
splits, streaming mode, column names, licensing). ``baligh.data.loader``
turns registry entries into ``load_dataset()`` calls; nothing else should
duplicate this information.
"""

from typing import Any

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

DATASETS: dict[str, dict[str, Any]] = {
    "arabicweb24": {
        "path": "lightonai/ArabicWeb24",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens": 28000000000,
        "domain": "web",
        "type": "cpt",
    },
    "arabictext_large": {
        "path": "Jr23xd23/ArabicText-Large",
        "split": "train",
        "streaming": False,
        "text_column": "text",
        "license": "Apache-2.0",
        "tokens": 1000000000,
        "domain": "general",
        "type": "cpt",
    },
    "arabic_pile": {
        "path": "premio-ai/TheArabicPile_Dialects",
        "name": "dedup",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens": 5000000000,
        "domain": "mixed",
        "type": "cpt",
    },
    "oscar_ar": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.ar",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens": 10000000000,
        "domain": "web",
        "type": "cpt",
    },
    "mc4_ar": {
        "path": "allenai/c4",
        "name": "ar",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens": 50000000000,
        "domain": "web",
        "type": "cpt",
    },
    "hadith_datasets": {
        "path": "meeAtif/hadith_datasets",
        "split": "train",
        "streaming": False,
        "text_column": "text",
        "license": "Open",
        "domain": "islamic",
        "type": "cpt_islamic",
    },
    "quran_qa": {
        "path": "nazimali/quran-question-answer-context",
        "split": "train",
        "streaming": False,
        "text_column": "context",
        "question_column": "question",
        "answer_column": "answer",
        "license": "CC",
        "domain": "islamic",
        "type": "cpt_islamic",
    },
    "quran_md": {
        "path": "Buraaq/quran-audio-text-dataset",
        "split": "train",
        "streaming": False,
        "text_column": "text",
        "license": "Open",
        "domain": "islamic",
        "type": "cpt_islamic",
    },
    "cidar": {
        "path": "arbml/CIDAR",
        "split": "train",
        "streaming": False,
        "instruction_column": "instruction",
        "input_column": "input",
        "output_column": "output",
        "license": "Apache-2.0",
        "domain": "instruction",
        "type": "sft",
    },
    "evol_instruct_arabic": {
        "path": "FreedomIntelligence/evol-instruct-arabic",
        "split": "train",
        "streaming": False,
        # Source ships ShareGPT turns (conversations/from/value), not flat
        # instruction columns; the standardizer branches on this flag.
        "format": "sharegpt",
        "license": "Open",
        "domain": "instruction",
        "type": "sft",
    },
    "summarization": {
        "path": "BounharAbdelaziz/Arabic-Synthetic-Summarization-Dataset",
        "split": "train",
        "streaming": False,
        # Source documents live in "text", targets in "summary"; the
        # standardizer below maps them onto the canonical instruction/output
        # schema before mixing/formatting.
        "instruction_column": "text",
        "input_column": None,
        "output_column": "summary",
        "license": "Open",
        "domain": "summarization",
        "type": "sft",
    },
    "islamweb_fatwa": {
        # Local artifact produced by src/scripts/convert_islamweb.py from
        # D:\AI\Datasets\Custom\islamweb1. Splits are pre-assigned by fatwa_id
        # so one ruling can never straddle train and eval.
        "path": {
            "train": r"D:\AI\datasets\baligh\islamweb_fatwa\train.jsonl",
            "eval": r"D:\AI\datasets\baligh\islamweb_fatwa\eval.jsonl",
        },
        "split": "train",
        "streaming": False,
        "instruction_column": "instruction",
        "input_column": "input",
        "output_column": "output",
        "license": "See source (Islamweb fatwa pages)",
        "domain": "islamic",
        "type": "sft",
    },
    "mmlu_arabic": {
        "path": "FreedomIntelligence/MMLU_Arabic",
        "split": "test",
        "streaming": False,
        "license": "Open",
        "domain": "eval",
        "type": "eval",
    },
    "cidar_eval": {
        "path": "arbml/CIDAR",
        "split": "test",
        "streaming": False,
        "license": "Apache-2.0",
        "domain": "eval",
        "type": "eval",
    },
    "mr_tydi_arabic": {
        "path": "castorini/mr-tydi",
        "name": "arabic",
        "split": "test",
        "streaming": False,
        "license": "Open",
        "domain": "eval",
        "type": "eval",
    },
}

# Canonical schemas produced by the standardizers
CPT_COLUMNS = ("text",)
SFT_COLUMNS = ("instruction", "input", "output")


def get_dataset_config(name: str) -> dict[str, Any]:
    """Return the registry entry for *name* (raises ValueError if unknown)."""
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset: {name}. Registered datasets: {known}")
    return DATASETS[name]


def list_datasets(dataset_type: str | None = None) -> dict[str, Any]:
    if dataset_type:
        return {k: v for k, v in DATASETS.items() if v.get("type") == dataset_type}
    return DATASETS


def get_cpt_datasets() -> dict[str, Any]:
    return list_datasets("cpt")


def get_sft_datasets() -> dict[str, Any]:
    return list_datasets("sft")


def get_eval_datasets() -> dict[str, Any]:
    return list_datasets("eval")


def get_islamic_datasets() -> dict[str, Any]:
    return list_datasets("cpt_islamic")


def _load_typed_datasets(
    dataset_type: str, split: str | None = None, streaming: bool | None = None
) -> dict[str, Any]:
    from baligh.data.loader import load_dataset_by_name

    result = {}
    for name, config in DATASETS.items():
        if config.get("type") != dataset_type:
            continue
        ds_split = split or config.get("split", "train")
        ds_streaming = streaming if streaming is not None else config.get("streaming", False)
        result[name] = load_dataset_by_name(name, split=ds_split, streaming=ds_streaming)
    return result


def load_cpt_datasets(split: str | None = None, streaming: bool | None = None) -> dict[str, Any]:
    return _load_typed_datasets("cpt", split=split, streaming=streaming)


def load_sft_datasets(split: str | None = None, streaming: bool | None = None) -> dict[str, Any]:
    return _load_typed_datasets("sft", split=split, streaming=streaming)


def load_eval_datasets(split: str | None = None, streaming: bool | None = None) -> dict[str, Any]:
    return _load_typed_datasets("eval", split=split, streaming=streaming)


def standardize_cpt_dataset(dataset: Any, name: str) -> Any:
    """Project any CPT source onto the canonical {"text"} schema."""
    config = get_dataset_config(name)
    text_column = config.get("text_column", "text")

    def _to_text(example: dict) -> dict:
        """Project a raw row onto the canonical text column."""
        return {"text": example.get(text_column) or ""}

    from datasets import IterableDataset

    kwargs: dict[str, Any] = {}
    if not isinstance(dataset, IterableDataset):
        # IterableDataset.map() does not support the `desc` argument.
        kwargs["desc"] = f"Standardize CPT: {name}"
    # Drop every original column: remove_columns affects only the OUTPUT
    # (the mapper still reads the full input), so retaining source columns
    # (id, metadata, ...) would leak non-canonical names into downstream
    # mix/clean steps and desync column lengths.
    kwargs["remove_columns"] = list(dataset.column_names)
    return dataset.map(_to_text, **kwargs)


def _standardize_sharegpt(dataset: Any, name: str, config: dict[str, Any]) -> Any:
    """Project a ShareGPT ``conversations`` source onto the canonical schema.

    ShareGPT rows carry ``[{"from": "human", "value": ...}, {"from": "gpt",
    "value": ...}]``. The first human turn becomes the instruction and the
    final gpt turn becomes the output; any turns between them are dropped
    rather than concatenated, because splicing unrelated turns into one
    completion teaches the model to answer two questions at once.
    """
    ins_key = config.get("sharegpt_human_key", "human")
    out_key = config.get("sharegpt_gpt_key", "gpt")

    def _to_canonical(example: dict) -> dict:
        turns = example.get("conversations") or []
        instruction = ""
        output = ""
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or turn.get("role") or ""
            value = turn.get("value") or turn.get("content") or ""
            if not instruction and role == ins_key:
                instruction = value
            if role == out_key and value:
                output = value
        return {"instruction": instruction, "input": "", "output": output}

    return dataset.map(
        _to_canonical,
        remove_columns=dataset.column_names,
        desc=f"Standardize SFT (ShareGPT): {name}",
    )


def standardize_sft_dataset(dataset: Any, name: str) -> Any:
    """Project any SFT source onto {"instruction", "input", "output"}.

    Column names come from the registry so sources like the summarization
    set (text -> instruction, summary -> output) map correctly instead of
    being silently dropped by the formatter. Sources declaring
    ``format: "sharegpt"`` take the conversational branch instead.
    """
    config = get_dataset_config(name)

    if config.get("format") == "sharegpt":
        return _standardize_sharegpt(dataset, name, config)

    ins_col = config.get("instruction_column", "instruction")
    in_col = config.get("input_column")
    out_col = config.get("output_column", "output")

    def _to_canonical(example: dict) -> dict:
        """Map registry columns onto instruction/input/output."""
        return {
            "instruction": example.get(ins_col) or "",
            "input": (example.get(in_col) or "") if in_col else "",
            "output": example.get(out_col) or "",
        }

    # Drop every original column: remove_columns applies to the OUTPUT only
    # (the mapper still reads the full input), so retaining source columns
    # here would leak non-canonical names like text/summary into the mix and
    # hand interleave_datasets a union schema. _to_canonical always emits
    # all three keys, so no add_column fallback is needed for missing input.
    return dataset.map(
        _to_canonical,
        remove_columns=list(dataset.column_names),
        desc=f"Standardize SFT: {name}",
    )
