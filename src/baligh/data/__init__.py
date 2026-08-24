"""Data package for Baligh-1.7B v0."""

from baligh.data.cleaner import CleaningPipeline, deduplicate_dataset, get_cleaning_pipeline
from baligh.data.datasets import (
    DATASETS,
    get_cpt_datasets,
    get_dataset_config,
    get_eval_datasets,
    get_islamic_datasets,
    get_sft_datasets,
    list_datasets,
    load_cpt_datasets,
    load_eval_datasets,
    load_sft_datasets,
    standardize_cpt_dataset,
    standardize_sft_dataset,
)
from baligh.data.formatter import PromptFormatter, get_cpt_formatter, get_sft_formatter
from baligh.data.loader import list_registered_names, load_dataset_by_name
from baligh.data.mixer import (
    DatasetMixer,
    get_cpt_mixer,
    get_sft_mixer,
    mix_cpt_datasets,
    mix_sft_datasets,
)
from baligh.data.validators import (
    check_dataset_schema,
    get_dataset_stats,
    validate_cpt_example,
    validate_dataset,
    validate_sft_example,
)

__all__ = [
    "load_dataset_by_name",
    "list_registered_names",
    "CleaningPipeline",
    "get_cleaning_pipeline",
    "deduplicate_dataset",
    "DatasetMixer",
    "get_cpt_mixer",
    "get_sft_mixer",
    "mix_cpt_datasets",
    "mix_sft_datasets",
    "PromptFormatter",
    "get_cpt_formatter",
    "get_sft_formatter",
    "validate_cpt_example",
    "validate_sft_example",
    "validate_dataset",
    "check_dataset_schema",
    "get_dataset_stats",
    "DATASETS",
    "get_dataset_config",
    "list_datasets",
    "get_cpt_datasets",
    "get_sft_datasets",
    "get_eval_datasets",
    "get_islamic_datasets",
    "load_cpt_datasets",
    "load_sft_datasets",
    "load_eval_datasets",
    "standardize_cpt_dataset",
    "standardize_sft_dataset",
]
