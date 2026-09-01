import pytest
import torch

from lloca.framesnet.frames import (
    ChangeOfFrames,
    Frames,
    IndexSelectFrames,
    InverseFrames,
    LowerIndicesFrames,
)
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import REPS, TOLERANCES


@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("reps", REPS)
def test_equivariance(batch_dims, reps):
    dtype = torch.float64

    reps = TensorReps(reps)
    trafo = TensorRepsTransform(TensorReps(reps))

    transform = rand_lorentz(batch_dims, dtype=dtype)
    frames = Frames(transform)

    x = torch.randn(*batch_dims, reps.dim, dtype=dtype)

    # manual transform
    transform_direct = torch.einsum(
        "...ij,...jk->...ik", frames.matrices, InverseFrames(frames).matrices
    )
    change_frames1 = Frames(transform_direct)
    x_prime1 = trafo(x, change_frames1)
    torch.testing.assert_close(x, x_prime1, **TOLERANCES)

    # all-in-one transform
    change_frames2 = ChangeOfFrames(frames, frames)
    x_prime2 = trafo(x, change_frames2)
    torch.testing.assert_close(x, x_prime2, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10]])
def test_change_of_frames_is_global(batch_dims):
    """Identity frames take the identity shortcut and must keep the (..., 4, 4) shape.

    This is the branch used by LLoCaMessagePassing with IdentityFrames / global
    RandomFrames; passing the full ``Frames.shape`` here used to produce (..., 4, 4, 4, 4).
    """
    fm = torch.randn(*batch_dims, 4, dtype=torch.float64).abs()
    frames = IdentityFrames()(fm)
    assert frames.is_global

    change = ChangeOfFrames(frames, frames)
    assert change.is_identity
    assert change.matrices.shape == (*batch_dims, 4, 4)
    assert change.inv.shape == (*batch_dims, 4, 4)
    assert change.det.shape == tuple(batch_dims)

    # a change from a frame to itself is the identity, so vectors are untouched
    trafo = TensorRepsTransform(TensorReps("1x1n"))
    torch.testing.assert_close(trafo(fm, change), fm, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10]])
def test_change_of_frames_between_two_global_frames(batch_dims):
    """Two *different* global frames still have a non-trivial change of frames.

    The identity shortcut used to trigger on ``frames_start.is_global`` alone, silently
    returning the identity for the bipartite (tuple) case in LLoCaMessagePassing.
    """
    dtype = torch.float64
    start = Frames(rand_lorentz([1], dtype=dtype).expand(*batch_dims, 4, 4), is_global=True)
    end = Frames(rand_lorentz([1], dtype=dtype).expand(*batch_dims, 4, 4), is_global=True)

    change = ChangeOfFrames(start, end)
    assert not change.is_identity
    assert change.is_global  # constant over particles, so still global
    torch.testing.assert_close(change.matrices, end.matrices @ start.inv, **TOLERANCES)
    eye = torch.eye(4, dtype=dtype).expand(*batch_dims, 4, 4)
    torch.testing.assert_close(change.matrices @ change.inv, eye, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10]])
def test_change_of_frames_index_selected_global(batch_dims):
    """Index selections of one global frames object still take the identity shortcut.

    Global frames are the same for every particle, so index selection cannot tell them
    apart; this is the branch LLoCaMessagePassing hits for global RandomFrames, and
    losing the shortcut here would transform every message for nothing.
    """
    dtype = torch.float64
    frames = Frames(rand_lorentz([1], dtype=dtype).expand(*batch_dims, 4, 4), is_global=True)
    idx_i = torch.arange(batch_dims[0])
    idx_j = idx_i.flip(0)

    change = ChangeOfFrames(IndexSelectFrames(frames, idx_j), IndexSelectFrames(frames, idx_i))
    assert change.is_identity

    # two selections of two *different* global frames must not take the shortcut
    other = Frames(rand_lorentz([1], dtype=dtype).expand(*batch_dims, 4, 4), is_global=True)
    change = ChangeOfFrames(IndexSelectFrames(frames, idx_j), IndexSelectFrames(other, idx_i))
    assert not change.is_identity
    torch.testing.assert_close(
        change.matrices, other.matrices[idx_i] @ frames.inv[idx_j], **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("is_identity", [True, False])
def test_lower_indices_frames(batch_dims, is_identity):
    """Lowering the indices multiplies by the metric, also for identity frames.

    Identity frames used to be passed through unchanged, because ``is_identity`` was
    propagated and ``TensorRepsTransform`` skips the transform for identity frames --
    so vector channels contracted euclidean instead of Minkowski.
    """
    dtype = torch.float64
    metric = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=dtype))
    eye = torch.eye(4, dtype=dtype).expand(*batch_dims, 4, 4)

    if is_identity:
        frames = Frames(is_identity=True, shape=tuple(batch_dims), device="cpu", dtype=dtype)
    else:
        frames = Frames(rand_lorentz(batch_dims, dtype=dtype))

    lowered = LowerIndicesFrames(frames)

    # lowering the indices is never the identity, so the transform must not be skipped
    assert not lowered.is_identity
    torch.testing.assert_close(lowered.matrices, metric @ frames.matrices, **TOLERANCES)
    torch.testing.assert_close(lowered.det, -frames.det, **TOLERANCES)
    torch.testing.assert_close(lowered.matrices @ lowered.inv, eye, **TOLERANCES)

    # a vector transformed with the lowered frames has its index lowered
    trafo = TensorRepsTransform(TensorReps("1x1n"))
    x = torch.randn(*batch_dims, 4, dtype=dtype)
    expected = torch.einsum("...ij,...j->...i", lowered.matrices, x)
    torch.testing.assert_close(trafo(x, lowered), expected, **TOLERANCES)
