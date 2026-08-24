"""Dataset mixing for Baligh-1.7B v0."""

from typing import Any

from datasets import interleave_datasets

from baligh.config import get_cpt_config, get_sft_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetMixer:
    def __init__(
        self,
        mix_config: dict[str, float],
        seed: int = 42,
        stopping_strategy: str = "first_exhausted",
        strict: bool = False,
    ):
        """Create a mixer.

        Args:
            mix_config: Mapping of dataset name -> mixing ratio.
            seed: Sampling seed for interleave.
            stopping_strategy: Passed to ``interleave_datasets``.
            strict: If True, raise on mix keys with no matching dataset.
                Production pipelines should use strict=True so that declared
                ratios can never silently drift from the trained mixture.
        """
        self.mix_config = dict(mix_config)
        self.seed = seed
        self.stopping_strategy = stopping_strategy
        self.strict = strict

    def mix(self, datasets: dict[str, Any]) -> Any:
        ordered_datasets = []
        ordered_probs = []
        missing = []
        total = sum(self.mix_config.values())
        for name, ratio in self.mix_config.items():
            if name in datasets:
                ordered_datasets.append(datasets[name])
                ordered_probs.append(ratio / total)
            else:
                missing.append(name)
        if missing:
            if self.strict:
                raise ValueError(
                    f"Mix config references unknown datasets: {sorted(missing)}. "
                    f"Provided: {sorted(datasets)}. Fix the registry or the mix config."
                )
            logger.warning(f"Datasets not found, skipping: {sorted(missing)}")
        if not ordered_datasets:
            raise ValueError("No valid datasets to mix")
        used = {name for name in self.mix_config if name in datasets}
        normalized = {name: round(self.mix_config[name] / total, 4) for name in used}
        logger.info(f"Mixing datasets with normalized ratios: {normalized}")
        return interleave_datasets(
            ordered_datasets,
            probabilities=ordered_probs,
            seed=self.seed,
            stopping_strategy=self.stopping_strategy,
        )


def get_cpt_mixer(seed: int = 42, strict: bool = False) -> DatasetMixer:
    config = get_cpt_config()
    return DatasetMixer(config.dataset_mix, seed=seed, strict=strict)


def get_sft_mixer(seed: int = 42, strict: bool = False) -> DatasetMixer:
    config = get_sft_config()
    return DatasetMixer(config.dataset_mix, seed=seed, strict=strict)


def mix_cpt_datasets(datasets: dict[str, Any], seed: int = 42, strict: bool = False) -> Any:
    mixer = get_cpt_mixer(seed, strict=strict)
    return mixer.mix(datasets)


def mix_sft_datasets(datasets: dict[str, Any], seed: int = 42, strict: bool = False) -> Any:
    mixer = get_sft_mixer(seed, strict=strict)
    return mixer.mix(datasets)
