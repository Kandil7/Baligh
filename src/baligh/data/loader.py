"""Dataset loading for Baligh-1.5B v0."""

from datasets import load_dataset

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

# Dataset registry with metadata
DATASET_REGISTRY = {
    "arabicweb24": {
        "path": "lightonai/ArabicWeb24",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens_estimate": 28000000000,
        "domain": "web",
    },
    "arabictext_large": {
        "path": "Jr23xd23/ArabicText-Large",
        "split": "train",
        "streaming": False,
        "text_column": "text",
        "license": "Apache-2.0",
        "tokens_estimate": 1000000000,
        "domain": "general",
    },
    "arabic_pile": {
        "path": "premio-ai/TheArabicPile_Dialects",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "license": "Open",
        "tokens_estimate": 5000000000,
        "domain": "mixed",
    },
}


def load_dataset_by_name(name: str, split: str = None, streaming: bool = None, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    config = DATASET_REGISTRY[name].copy()
    split = split or config.pop("split", "train")
    streaming = streaming if streaming is not None else config.pop("streaming", False)
    path = config.pop("path")
    name_param = config.pop("name", None)
    logger.info(f"Loading dataset: {name} (split={split}, streaming={streaming})")
    return load_dataset(path, name=name_param, split=split, streaming=streaming, **config, **kwargs)
