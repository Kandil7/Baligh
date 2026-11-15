"""Dataset loading for Baligh-1.7B v0.

Single source of truth for dataset metadata lives in
:mod:`baligh.data.datasets` (``DATASETS``). This module only knows how to
turn a registry entry into a Hugging Face ``load_dataset()`` call.

Registry entries may point at a local JSON/JSONL artifact instead of a Hub
repo id (see ``islamweb_fatwa``). Local sources are detected by the path
existing on disk, or by ``path`` being a mapping of split -> file, which is
how a pre-split source keeps its splits apart.
"""

from pathlib import Path
from typing import Any

from datasets import load_dataset

from baligh.data.datasets import DATASETS, get_dataset_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

_LOCAL_SUFFIXES = (".json", ".jsonl")


def list_registered_names(dataset_type: str | None = None) -> list[str]:
    """Return names of registered datasets, optionally filtered by type."""
    return [
        name
        for name, cfg in DATASETS.items()
        if dataset_type is None or cfg.get("type") == dataset_type
    ]


def _is_local_source(path: Any) -> bool:
    """True when *path* names a local artifact rather than a Hub repo id.

    A mapping is always local (split -> file). A string is local when it has
    a JSON suffix AND resolves on disk, so a Hub id like "arbml/CIDAR" is
    never mistaken for a file.
    """
    if isinstance(path, dict):
        return True
    if not isinstance(path, str):
        return False
    if not path.lower().endswith(_LOCAL_SUFFIXES):
        return False
    return Path(path).expanduser().exists()


def _load_local(path: Any, split: str, streaming: bool, **kwargs: Any) -> Any:
    """Load a local JSON/JSONL source as a ``datasets`` object.

    A split mapping selects the file for *split*; a bare string is loaded as
    whatever split the caller asked for (``load_dataset`` labels a single
    ``data_files`` payload "train", so the requested split is aliased onto it).
    """
    if isinstance(path, dict):
        if split not in path:
            known = ", ".join(sorted(path))
            raise ValueError(f"Local source has no split '{split}'. Available splits: {known}")
        data_files: Any = {split: path[split]}
    else:
        data_files = {split: path}

    return load_dataset(
        "json",
        data_files=data_files,
        split=split,
        streaming=streaming,
        **kwargs,
    )


def load_dataset_by_name(
    name: str,
    split: str | None = None,
    streaming: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Load a dataset from the central registry by name.

    Args:
        name: Registry key (see ``baligh.data.datasets.DATASETS``).
        split: Override the registered split.
        streaming: Override the registered streaming mode.
        **kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        The loaded dataset (Dataset or IterableDataset).
    """
    config = get_dataset_config(name)  # raises ValueError for unknown names

    split = split or config.get("split", "train")
    streaming = streaming if streaming is not None else bool(config.get("streaming", False))
    path = config["path"]
    name_param = config.get("name")

    if _is_local_source(path):
        logger.info(f"Loading local dataset: {name} (path={path}, split={split})")
        return _load_local(path, split=split, streaming=streaming, **kwargs)

    logger.info(f"Loading dataset: {name} (path={path}, split={split}, streaming={streaming})")
    return load_dataset(
        path,
        name=name_param,
        split=split,
        streaming=streaming,
        **kwargs,
    )
