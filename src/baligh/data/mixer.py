"""Dataset mixing for Baligh-1.5B v0."""

from datasets import Dataset, interleave_datasets
from baligh.config import get_cpt_config, get_sft_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

class DatasetMixer:
    def __init__(self, mix_config, seed=42, stopping_strategy='first_exhausted'):
        self.mix_config = mix_config
        self.seed = seed
        self.stopping_strategy = stopping_strategy
    
    def mix(self, datasets):
        ordered_datasets = []
        ordered_probs = []
        total = sum(self.mix_config.values())
        for name, ratio in self.mix_config.items():
            if name in datasets:
                ordered_datasets.append(datasets[name])
                ordered_probs.append(ratio / total)
            else:
                logger.warning('Dataset %s not found, skipping' % name)
        if not ordered_datasets:
            raise ValueError('No valid datasets to mix')
        logger.info('Mixing datasets with ratios: %s' % dict(zip([str(d) for d in ordered_datasets], ordered_probs)))
        return interleave_datasets(ordered_datasets, probabilities=ordered_probs, seed=self.seed, stopping_strategy=self.stopping_strategy)

def get_cpt_mixer(seed=42):
    config = get_cpt_config()
    return DatasetMixer(config.dataset_mix, seed=seed)

def get_sft_mixer(seed=42):
    config = get_sft_config()
    return DatasetMixer(config.dataset_mix, seed=seed)

def mix_cpt_datasets(datasets, seed=42):
    mixer = get_cpt_mixer(seed)
    return mixer.mix(datasets)

def mix_sft_datasets(datasets, seed=42):
    mixer = get_sft_mixer(seed)
    return mixer.mix(datasets)
