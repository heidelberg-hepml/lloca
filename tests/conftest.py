"""Shared pytest configuration."""

import pytest
import torch


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed torch before every test.

    The equivariance tests compare two numerically sensitive paths at tight tolerances,
    and several code paths draw from the global RNG (e.g. the lightlike/coplanar
    regularization in ``orthogonalize_4d``). Seeding makes a failure reproducible from the
    test id alone.
    """
    torch.manual_seed(42)
