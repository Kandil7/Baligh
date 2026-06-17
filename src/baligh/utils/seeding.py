"""Random seeding utilities for reproducibility."""

import os
import random

import numpy as np
import torch

from baligh.config import get_config


def set_seed(seed: int | None = None) -> int:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed. If None, uses config default from config default.

    Returns:
        The seed that was set.
    """
    if seed is None:
        config = get_config()
        seed = config.seed

    # Python built-in
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch deterministic settings
    config = get_config()
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """Worker init function for DataLoader seeding."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator() -> torch.Generator:
    """Get a torch Generator with the configured seed."""
    config = get_config()
    g = torch.Generator()
    g.manual_seed(config.seed)
    return g
