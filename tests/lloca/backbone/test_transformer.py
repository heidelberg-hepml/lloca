import pytest
import torch

from lloca.backbone.transformer import Transformer
from lloca.backbone.transformer_v2 import Transformer as TransformerV2
from lloca.framesnet.equi_frames import LearnedPDFrames
from lloca.framesnet.frames import InverseFrames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import FRAMES_PREDICTOR, LOGM2_MEAN_STD, REPS, TOLERANCES
from tests.helpers import equivectors_builder, sample_particle


@pytest.mark.parametrize("transformer_type", [Transformer, TransformerV2])
@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("attn_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_transformer_invariance_equivariance(
    transformer_type,
    FramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
    attn_reps,
    num_blocks=2,
    num_heads=2,
):
    dtype = torch.float64

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    predictor.equivectors.init_standardization(fm_test)

    # define transformer
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    net = transformer_type(
        in_channels=in_reps.dim,
        attn_reps=attn_reps,
        out_channels=in_reps.dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(dtype=dtype)

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = predictor(fm)
    fm_local = trafo(fm, frames)

    # global - transformer
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    fm_tr_prime_local = net(fm_tr_local, frames_transformed)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseFrames(frames_transformed))

    # transformer - global
    fm_prime_local = net(fm_local, frames)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)


@pytest.mark.parametrize("transformer_type", [Transformer, TransformerV2])
@pytest.mark.parametrize("FramesPredictor", [IdentityFrames, LearnedPDFrames])
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("attn_reps", [REPS[-1]])
@pytest.mark.parametrize("logm2_mean,logm2_std", [LOGM2_MEAN_STD[0]])
@pytest.mark.parametrize(
    "checkpoint_blocks,compile", [(False, False), (True, False), (False, True)]
)
def test_transformer_shape(
    transformer_type,
    FramesPredictor,
    batch_dims,
    attn_reps,
    logm2_mean,
    logm2_std,
    checkpoint_blocks,
    compile,
    num_blocks=2,
    num_heads=2,
):
    kwargs = {}
    if FramesPredictor == LearnedPDFrames:
        kwargs["equivectors"] = equivectors_builder()
    predictor = FramesPredictor(**kwargs)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean)
    if FramesPredictor == LearnedPDFrames:
        predictor.equivectors.init_standardization(fm_test)

    # define transformer
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    net = transformer_type(
        in_channels=in_reps.dim,
        attn_reps=attn_reps,
        out_channels=in_reps.dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        checkpoint_blocks=checkpoint_blocks,
        compile=compile,
    )

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean)
    frames = predictor(fm)
    fm_local = trafo(fm, frames)

    # call transformer
    out = net(fm_local, frames)
    assert out.shape == (*batch_dims, 4)
