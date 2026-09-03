import pytest
import torch
from torch.nn import Linear

from lloca.backbone.attention import LLoCaAttention
from lloca.framesnet.equi_frames import LearnedPDFrames
from lloca.framesnet.frames import Frames, InverseFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import FRAMES_PREDICTOR, LOGM2_MEAN_STD, REPS, TOLERANCES
from tests.helpers import equivectors_builder, sample_particle


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    hidden_reps,
    logm2_std,
    logm2_mean,
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)

    def call_predictor(fm):
        return predictor(fm)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    predictor.equivectors.init_standardization(fm_test)

    # preparations
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(hidden_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    attention = LLoCaAttention(hidden_reps, 1).to(dtype=dtype)
    linear_in = Linear(in_reps.dim, 3 * hidden_reps.dim).to(dtype=dtype)
    linear_out = Linear(hidden_reps.dim, in_reps.dim).to(dtype=dtype)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # path 1: Frames transform + random transform
    frames = call_predictor(fm)
    fm_local = trafo(fm, frames)
    attention.prepare_frames(frames, p_ref=fm.sum(dim=-2))
    x_local = linear_in(fm_local).unsqueeze(0)
    q_local, k_local, v_local = x_local.chunk(3, dim=-1)
    x_local2 = attention(q_local, k_local, v_local).squeeze(0)
    fm_local = linear_out(x_local2)
    fm_global = trafo(fm_local, InverseFrames(frames))
    fm_global_prime = torch.einsum("...ij,...j->...i", random, fm_global)

    # path 2: random transform + Frames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    frames_prime = call_predictor(fm_prime)
    fm_prime_local = trafo(fm_prime, frames_prime)
    attention.prepare_frames(frames_prime, p_ref=fm_prime.sum(dim=-2))
    x_prime_local = linear_in(fm_prime_local).unsqueeze(0)
    q_prime_local, k_prime_local, v_prime_local = x_prime_local.chunk(3, dim=-1)
    x_prime_local2 = attention(q_prime_local, k_prime_local, v_prime_local).squeeze(0)
    fm_prime_local = linear_out(x_prime_local2)
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames_prime))

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_prime_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_local2, x_prime_local2, **TOLERANCES)

    # test equivariance of output
    torch.testing.assert_close(fm_prime_global, fm_global_prime, **TOLERANCES)


def _frames_and_momenta(n=10, dtype=torch.float64):
    equivectors = equivectors_builder()
    predictor = LearnedPDFrames(equivectors=equivectors).to(dtype=dtype)
    fm = sample_particle([n], 1.0, 0.0, dtype=dtype)
    predictor.equivectors.init_standardization(fm)
    return predictor(fm), fm


def test_preserve_variance_requires_p_ref():
    """``preserve_variance`` needs a reference momentum and must say so."""
    frames, _ = _frames_and_momenta()
    attention = LLoCaAttention(TensorReps("4x0n+2x1n"), 1).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="p_ref"):
        attention.prepare_frames(frames)


def test_preserve_variance_off_ignores_p_ref():
    """With the flag off, ``p_ref`` is not required and not used."""
    frames, fm = _frames_and_momenta()
    attention = LLoCaAttention(TensorReps("4x0n+2x1n"), 1, preserve_variance=False).to(
        dtype=torch.float64
    )

    attention.prepare_frames(frames)  # must not raise
    qkv_without = attention.frames_qkv.matrices.clone()

    attention.prepare_frames(frames, p_ref=fm.sum(dim=-2))
    torch.testing.assert_close(attention.frames_qkv.matrices, qkv_without, **TOLERANCES)


def test_preserve_variance_packed_matches_dense():
    """The packed (``ptr``) branch of _compute_gamma must agree with the dense one."""
    dtype = torch.float64
    frames, fm = _frames_and_momenta(n=10, dtype=dtype)
    attention = LLoCaAttention(TensorReps("4x0n+2x1n"), 1).to(dtype=dtype)

    # dense: one event of 10 particles
    attention.prepare_frames(frames, p_ref=fm.sum(dim=-2))
    dense = attention.frames_qkv.matrices.clone()

    # packed: the same 10 particles as a single jet described by ptr
    ptr = torch.tensor([0, 10])
    attention.prepare_frames(frames, p_ref=fm.sum(dim=-2, keepdim=True), ptr=ptr)
    torch.testing.assert_close(attention.frames_qkv.matrices, dense, **TOLERANCES)


def test_preserve_variance_bounds_boosted_variance():
    """The point of the flag: a hard boost must not blow up the transformed q/k/v."""
    dtype = torch.float64
    frames, fm = _frames_and_momenta(n=64, dtype=dtype)
    reps = TensorReps("4x0n+2x1n")
    x = torch.randn(1, 1, 64, reps.dim, dtype=dtype)

    scales = {}
    for preserve in (False, True):
        attention = LLoCaAttention(reps, 1, preserve_variance=preserve).to(dtype=dtype)
        attention.prepare_frames(frames, p_ref=fm.sum(dim=-2))
        q, k, v = attention._local_to_global(x, x, x)
        scales[preserve] = q.abs().max().item()

    assert scales[True] < scales[False], (
        f"preserve_variance did not reduce the global-frame scale: {scales}"
    )


@pytest.mark.parametrize("order", [1, 2])
def test_parity_odd_matches_parity_even_for_proper_frames(order):
    """All Frames-Net classes predict proper frames, so ``Xp`` must behave like ``Xn`` there.

    Regression test: ``LowerIndicesFrames`` multiplies by the metric, whose determinant is -1.
    If that leaked into the parity factor, the keys would be negated relative to the queries
    and the parity-odd channels would flip the sign of their contribution to the logits.
    """
    dtype = torch.float64
    frames, fm = _frames_and_momenta(dtype=dtype)
    torch.testing.assert_close(
        frames.det, torch.ones_like(frames.det), **TOLERANCES
    )  # proper frames

    outputs = []
    for parity in ("n", "p"):
        attention = LLoCaAttention(TensorReps(f"4x0n+2x{order}{parity}"), 1).to(dtype=dtype)
        attention.prepare_frames(frames, p_ref=fm.sum(dim=-2))
        torch.manual_seed(0)
        qkv = torch.randn(1, 1, fm.shape[0], attention.transform.reps.dim, dtype=dtype)
        outputs.append(attention(qkv, qkv, qkv))

    torch.testing.assert_close(outputs[0], outputs[1], **TOLERANCES)


@pytest.mark.parametrize("order", [1, 2])
def test_parity_odd_flips_under_improper_frames(order):
    """A parity-odd rep must still pick up sign(det L) when the frames are improper."""
    dtype = torch.float64
    reps = TensorReps(f"2x{order}p")
    trafo = TensorRepsTransform(reps)

    proper = rand_lorentz([10], dtype=dtype)
    parity_flip = torch.eye(4, dtype=dtype)
    parity_flip[1, 1] = -1
    improper = Frames(proper @ parity_flip)

    x = torch.randn(10, reps.dim, dtype=dtype)
    expected = -TensorRepsTransform(TensorReps(f"2x{order}n"))(x, improper)
    torch.testing.assert_close(trafo(x, improper), expected, **TOLERANCES)
