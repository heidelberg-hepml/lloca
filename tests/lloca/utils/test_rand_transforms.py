import pytest
import torch

from lloca.utils.rand_transforms import (
    rand_boost,
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_ztransform,
    sample_rapidity,
)
from tests.constants import BATCH_DIMS, MILD_TOLERANCES, TOLERANCES
from tests.helpers import lorentz_test


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("std_eta", [0.1, 1, 2])
@pytest.mark.parametrize(
    "transform_type",
    [
        rand_lorentz,
        rand_rotation,
        rand_xyrotation,
        rand_ztransform,
        rand_boost,
    ],
)
def test_rand_transform(batch_dims, std_eta, transform_type):
    dtype = torch.float64

    # collect N different kinds of transformations
    kwargs = {
        "shape": batch_dims,
        "dtype": dtype,
    }
    if transform_type in [rand_lorentz, rand_ztransform, rand_boost]:
        kwargs["std_eta"] = std_eta
    transform = transform_type(**kwargs)

    # test that this is a valid Lorentz transform
    lorentz_test(transform, **MILD_TOLERANCES)

    # test that batch entries are actually distinct (catches indexing bugs
    # that would silently collapse all batches to the same transform)
    flat = transform.reshape(-1, 4, 4)
    assert flat.shape[0] == 1 or not torch.allclose(flat[0], flat[-1])

    # boosts must stay subluminal
    if transform_type in [rand_lorentz, rand_ztransform, rand_boost]:
        beta = (transform[..., 0, 1:] / transform[..., 0, 0].unsqueeze(-1)).norm(dim=-1)
        assert beta.max() < 1, f"superluminal boost: |beta|={beta.max()}"

    # test specific properties of the transform
    if transform_type in [rand_rotation, rand_xyrotation]:
        should_zero = torch.cat([transform[..., 0, 1:].flatten(), transform[..., 1:, 0].flatten()])
        if transform_type == rand_xyrotation:
            should_zero = torch.cat(
                [
                    should_zero,
                    transform[..., 3, 1:2].flatten(),
                    transform[..., 1:2, 3].flatten(),
                ]
            )
        torch.testing.assert_close(should_zero, torch.zeros_like(should_zero), **TOLERANCES)


@pytest.mark.parametrize("std_eta", [0.1, 1.0])
@pytest.mark.parametrize("n_max_std_eta", [1.0, 3.0])
def test_sample_rapidity_is_truncated(std_eta, n_max_std_eta):
    """``n_max_std_eta`` must actually truncate; ``clamp`` is not in-place."""
    eta = sample_rapidity((200000,), std_eta=std_eta, n_max_std_eta=n_max_std_eta)
    bound = std_eta * n_max_std_eta
    assert eta.abs().max() <= bound + 1e-9

    # and the bound must be attained, i.e. we are not truncating far too aggressively
    assert eta.abs().max() > 0.9 * bound
