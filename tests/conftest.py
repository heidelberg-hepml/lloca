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

    It also pins the discrete part of these comparisons: ParticleNet's kNN graph is invariant
    only up to tie-breaking, so for a minority of seeds a near-tied distance selects a
    different neighbour on the two paths and moves the score past the tolerance. See the
    comment on the score assertion in ``test_particlenet.py``.
    """
    torch.manual_seed(42)
