"""Dataset loading for Baligh-1.7B v0.

Single source of truth for dataset metadata lives in
:mod:`baligh.data.datasets` (``DATASETS``). This module only knows how to
turn a registry entry into a Hugging Face ``load_dataset()`` call.
"""

from typing import Any

from datasets import load_dataset

from baligh.data.datasets import DATASETS, get_dataset_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def list_registered_names(dataset_type: str | None = None) -> list[str]:
    """Return names of registered datasets, optionally filtered by type."""
    return [
        name
        for name, cfg in DATASETS.items()
        if dataset_type is None or cfg.get("type") == dataset_type
    ]


def load_dataset_by_name(
    name: str,
    split: str | None = None,
    streaming: bool | None = None,
    **kwargs: Any,
):
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

    logger.info(f"Loading dataset: {name} (path={path}, split={split}, streaming={streaming})")
    return load_dataset(
        path,
        name=name_param,
        split=split,
        streaming=streaming,
        **kwargs,
    )
