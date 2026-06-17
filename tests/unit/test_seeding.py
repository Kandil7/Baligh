"""Tests for seeding utilities."""

import random

import numpy as np
import torch

from baligh.utils.seeding import get_generator, seed_worker, set_seed


class TestSetSeed:
    def test_sets_seed(self):
        set_seed(42)
        a1 = random.random()
        set_seed(42)
        a2 = random.random()
        assert a1 == a2

    def test_sets_numpy_seed(self):
        set_seed(42)
        a1 = np.random.rand()
        set_seed(42)
        a2 = np.random.rand()
        assert a1 == a2

    def test_sets_torch_seed(self):
        set_seed(42)
        a1 = torch.rand(1).item()
        set_seed(42)
        a2 = torch.rand(1).item()
        assert a1 == a2

    def test_returns_seed(self):
        result = set_seed(42)
        assert result == 42

    def test_default_seed(self):
        result = set_seed()
        assert isinstance(result, int)


class TestGetGenerator:
    def test_returns_generator(self):
        gen = get_generator()
        assert isinstance(gen, torch.Generator)

    def test_deterministic(self):
        g1 = get_generator()
        g2 = get_generator()
        v1 = torch.rand(1, generator=g1).item()
        v2 = torch.rand(1, generator=g2).item()
        assert v1 == v2


class TestSeedWorker:
    def test_does_not_crash(self):
        seed_worker(0)
        seed_worker(1)
