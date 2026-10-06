import math

import pytest
import torch
from torch.nn import Linear

from lloca.backbone.attention import LLoCaAttention
from lloca.framesnet.equi_frames import LearnedPDFrames
from lloca.framesnet.frames import Frames, InverseFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.polar_decomposition import restframe_boost
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import FRAMES_PREDICTOR, LOGM2_MEAN_STD, REPS, STRICT_TOLERANCES, TOLERANCES
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


@pytest.mark.parametrize("kwargs", [{}, dict(preserve_variance=False, lightcone=True)])
def test_requires_p_ref(kwargs):
    """``preserve_variance`` and ``lightcone`` need a reference momentum and must say so."""
    frames, _ = _frames_and_momenta()
    attention = LLoCaAttention(TensorReps("4x0n+2x1n"), 1, **kwargs).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="p_ref"):
        attention.prepare_frames(frames)


def test_global_frames_lightcone_needs_no_p_ref():
    """Global frames fall back to standard attention, so ``lightcone`` needs no ``p_ref``."""
    frames = Frames(is_identity=True, shape=(10,), device="cpu", dtype=torch.float64)
    attention = LLoCaAttention(TensorReps("4x0n+2x1n"), 1, lightcone=True)
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


@pytest.mark.parametrize("lightcone", [False, True])
def test_packed_matches_dense(lightcone):
    """The packed (``ptr``) layout with several jets agrees with separate dense calls per jet."""
    dtype = torch.float64
    sizes = [4, 7, 5]
    reps = TensorReps("2x0n+2x1n+1x2n+1x1p")
    attention = LLoCaAttention(reps, 2, lightcone=lightcone).to(dtype=dtype)
    jets = [_frames_and_momenta(n=n, dtype=dtype) for n in sizes]
    qkv = [torch.randn(1, 2, sum(sizes), reps.dim, dtype=dtype) for _ in range(3)]

    dense = []
    for (frames, fm), *x in zip(jets, *(x.split(sizes, dim=-2) for x in qkv), strict=True):
        attention.prepare_frames(frames, p_ref=fm.sum(dim=-2))
        dense.append(attention(*x))

    matrices = torch.cat([frames.matrices.detach() for frames, _ in jets])
    ptr = torch.tensor([0, *sizes]).cumsum(0)
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    p_ref = torch.stack([fm.sum(dim=-2) for _, fm in jets])
    attention.prepare_frames(Frames(matrices), p_ref=p_ref, ptr=ptr)
    packed = attention(*qkv, attn_mask=batch[:, None] == batch[None, :])
    torch.testing.assert_close(packed, torch.cat(dense, dim=-2), **STRICT_TOLERANCES)


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


@pytest.mark.parametrize("preserve_variance", [False, True])
def test_lightcone_matches_cartesian(preserve_variance):
    """Light-cone coordinates are an exact change of basis: same outputs and gradients."""
    dtype = torch.float64
    frames, fm = _frames_and_momenta(dtype=dtype)
    matrices = frames.matrices.detach()
    reps = TensorReps("2x0n+2x1n+2x2n+2x1p")
    qkv = [torch.randn(1, 2, fm.shape[0], reps.dim, dtype=dtype) for _ in range(3)]

    results = []
    for lightcone in (False, True):
        attention = LLoCaAttention(
            reps, 2, preserve_variance=preserve_variance, lightcone=lightcone
        ).to(dtype=dtype)
        inputs = [x.clone().requires_grad_() for x in (matrices, *qkv)]
        attention.prepare_frames(Frames(inputs[0]), p_ref=fm.sum(dim=-2))
        outputs = attention(*inputs[1:])
        outputs.square().sum().backward()
        results.append((attention.frames_qkv.matrices, outputs, *(x.grad for x in inputs)))

    (qkv_frames, *cartesian), (qkv_frames_lc, *lightcone) = results
    assert not torch.allclose(qkv_frames, qkv_frames_lc)  # the option takes effect
    for x, x_lc in zip(cartesian, lightcone, strict=True):
        torch.testing.assert_close(x_lc, x, **STRICT_TOLERANCES)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
        ),
    ],
)
@pytest.mark.parametrize("lightcone", [False, True])
def test_lightcone_bfloat16_boosted(lightcone, device):
    """Under bfloat16 autocast, only light-cone coordinates keep a boosted jet accurate."""
    frames, fm = _frames_and_momenta(n=32)
    sinh, cosh = math.sinh(4.0), math.cosh(4.0)  # boost with rapidity 4
    boost = restframe_boost(torch.tensor([cosh, 0.6 * sinh, 0.0, 0.8 * sinh], dtype=torch.float64))
    matrices = (frames.matrices @ boost.inverse()).detach().to(device)
    p_ref = (fm @ boost.mT).sum(dim=-2).to(device)
    reps = TensorReps("4x0n+2x1n+1x2n")
    attention = LLoCaAttention(reps, 2, lightcone=lightcone).to(device)
    qkv = [torch.randn(1, 2, 32, reps.dim, device=device, dtype=torch.bfloat16) for _ in range(3)]

    attention.prepare_frames(Frames(matrices), p_ref=p_ref)
    reference = attention(*(x.double() for x in qkv))
    with torch.autocast(device, dtype=torch.bfloat16):
        # float32 frames as from the Frames-Net, bfloat16 q/k/v as from a linear layer
        attention.prepare_frames(Frames(matrices.float()), p_ref=p_ref.float())
        outputs = attention(*qkv)
    assert outputs.dtype == torch.bfloat16
    error = ((outputs.double() - reference).norm() / reference.norm()).item()
    assert (error < 2e-2) == lightcone


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
