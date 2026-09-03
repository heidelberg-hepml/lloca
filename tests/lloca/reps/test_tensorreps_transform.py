import pytest
import torch

from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dim", BATCH_DIMS)
def test_tensorreps(batch_dim):
    frames = rand_lorentz(batch_dim)
    frames = Frames(frames)

    rep = "5x0n+3x1n+3x2n+5x3n"
    rep = TensorReps(rep)
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    tensor_reps_transform(coeffs, frames)

    # 0n test
    rep = TensorReps("5x0n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(transformed_coeffs, coeffs, **TOLERANCES)

    # 0p test
    rep = TensorReps("5x0p")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(transformed_coeffs, coeffs * frames.det[..., None], **TOLERANCES)

    # 1n test
    rep = TensorReps("5x1n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
        torch.matmul(
            coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
            frames.matrices.transpose(-1, -2),
        ),
        **TOLERANCES,
    )

    # 1p test
    rep = TensorReps("5x1p")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
        torch.matmul(
            coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
            frames.matrices.transpose(-1, -2),
        )
        * frames.det[..., None, None],
        **TOLERANCES,
    )

    # 2n test
    rep = TensorReps("5x2n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    naive_trafo = torch.einsum(
        "...ij, ...lm, ...cjm -> ...cil",
        frames.matrices,
        frames.matrices,
        coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
    )
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )

    # 2p test
    rep = TensorReps("5x2p")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    naive_trafo = torch.einsum(
        "...ij, ...lm, ...cjm -> ...cil",
        frames.matrices,
        frames.matrices,
        coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
    )
    naive_trafo *= frames.det[..., None, None, None]
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )

    # 1x0n+1x2n test
    rep = TensorReps("1x0n+1x2n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    naive_trafo = torch.einsum(
        "...ij, ...lm, ...cjm -> ...cil",
        frames.matrices,
        frames.matrices,
        coeffs[..., 1:].reshape(*batch_dim, rep.max_rep.mul, 4, 4),
    )
    torch.testing.assert_close(
        transformed_coeffs[..., 1:].reshape(*batch_dim, rep.max_rep.mul, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )
    torch.testing.assert_close(coeffs[..., :1], transformed_coeffs[..., :1], **TOLERANCES)


@pytest.mark.parametrize("reps", ["1x1n+1x1p", "1x1p+1x1n", "2x0n+1x1n+1x1p", "1x2n+1x2p"])
@pytest.mark.parametrize("batch_dim", [[10]])
def test_tensorreps_mixed_parity(batch_dim, reps):
    """Reps of the same order but opposite parity are distinct and both have to be transformed.

    They used to share a single slot per order, so one of them was silently left untransformed
    (order 1) or dropped altogether (order 2).
    """
    dtype = torch.float64
    matrices = rand_lorentz(batch_dim, dtype=dtype)
    frames = Frames(matrices)
    sign = frames.det.sign()[..., None, None]

    reps = TensorReps(reps)
    coeffs = torch.randn(*batch_dim, reps.dim, dtype=dtype)
    transformed_coeffs = TensorRepsTransform(reps)(coeffs, frames)

    # transform each block on its own and compare
    idx = 0
    for mul, rep in reps:
        block = coeffs[..., idx : idx + mul * 4**rep.order]
        block = block.reshape(*batch_dim, mul, *(rep.order * (4,)))
        if rep.order == 1:
            block = torch.einsum("...ij,...cj->...ci", matrices, block)
        elif rep.order == 2:
            block = torch.einsum("...ij,...lm,...cjm->...cil", matrices, matrices, block)
        if rep.parity == -1:
            block = block * sign.reshape(*batch_dim, *(rep.order + 1) * (1,))
        torch.testing.assert_close(
            transformed_coeffs[..., idx : idx + mul * 4**rep.order].reshape(block.shape),
            block,
            **TOLERANCES,
        )
        idx += mul * 4**rep.order
