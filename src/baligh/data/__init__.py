"""Data package for Baligh-1.5B v0."""

from baligh.data.loader import load_dataset_by_name, load_cpt_datasets, load_sft_datasets, load_eval_datasets, interleave_datasets_by_ratio
from baligh.data.cleaner import CleaningPipeline, get_cleaning_pipeline, deduplicate_dataset
from baligh.data.mixer import DatasetMixer, get_cpt_mixer, get_sft_mixer, mix_cpt_datasets, mix_sft_datasets
from baligh.data.formatter import PromptFormatter, get_cpt_formatter, get_sft_formatter
from baligh.data.validators import validate_cpt_example, validate_sft_example, validate_dataset, check_dataset_schema, get_dataset_stats
from baligh.data.datasets import DATASETS, get_dataset_config, list_datasets, get_cpt_datasets, get_sft_datasets, get_eval_datasets, get_islamic_datasets

__all__ = [
    'load_dataset_by_name', 'load_cpt_datasets', 'load_sft_datasets', 'load_eval_datasets', 'interleave_datasets_by_ratio',
    'CleaningPipeline', 'get_cleaning_pipeline', 'deduplicate_dataset',
    'DatasetMixer', 'get_cpt_mixer', 'get_sft_mixer', 'mix_cpt_datasets', 'mix_sft_datasets',
    'PromptFormatter', 'get_cpt_formatter', 'get_sft_formatter',
    'validate_cpt_example', 'validate_sft_example', 'validate_dataset', 'check_dataset_schema', 'get_dataset_stats',
    'DATASETS', 'get_dataset_config', 'list_datasets', 'get_cpt_datasets', 'get_sft_datasets', 'get_eval_datasets', 'get_islamic_datasets',
]
