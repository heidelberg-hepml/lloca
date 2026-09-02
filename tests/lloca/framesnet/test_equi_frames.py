import pytest
import torch

from lloca.framesnet.equi_frames import (
    LearnedPDFrames,
    LearnedRestFrames,
    LearnedSO2Frames,
    LearnedSO3Frames,
    LearnedSO13Frames,
    LearnedZFrames,
    average_event,
)
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_squarednorm
from lloca.utils.rand_transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_ztransform,
)
from tests.constants import LOGM2_MEAN_STD, TOLERANCES
from tests.helpers import equivectors_builder, lorentz_test, sample_particle, sweep

PREDICTORS = [
    (LearnedSO13Frames, rand_lorentz),
    (LearnedRestFrames, rand_lorentz),
    (LearnedPDFrames, rand_lorentz),
    (LearnedSO3Frames, rand_rotation),
    (LearnedZFrames, rand_ztransform),
    (LearnedSO2Frames, rand_xyrotation),
]

SWEEP = sweep(
    dict(
        FramesPredictor=LearnedSO13Frames,
        rand_trafo=rand_lorentz,
        logm2_mean=0,
        logm2_std=1,
        predictor_kwargs={},
    ),
    ("FramesPredictor,rand_trafo", PREDICTORS),
    ("logm2_mean,logm2_std", LOGM2_MEAN_STD),
    ("predictor_kwargs", [dict(fix_params=True)]),
    # the boost regularization is equivariant, so it belongs in these sweeps; it only exists on
    # the predictors that produce a boost, with both the hard and the soft clamp
    (
        "FramesPredictor,rand_trafo,predictor_kwargs",
        [
            (LearnedPDFrames, rand_lorentz, dict(gamma_max=10.0)),
            (LearnedPDFrames, rand_lorentz, dict(gamma_max=10.0, gamma_hardness=2.0)),
            (LearnedZFrames, rand_ztransform, dict(gamma_max=10.0)),
        ],
    ),
)


@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize(*SWEEP)
def test_frames_transformation(
    FramesPredictor, rand_trafo, batch_dims, logm2_std, logm2_mean, predictor_kwargs
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors, **predictor_kwargs).to(dtype=dtype)

    def call_predictor(fm):
        return predictor(fm)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    predictor.equivectors.init_standardization(fm_test)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # frames for un-transformed fm
    frames = call_predictor(fm)
    lorentz_test(frames.matrices, **TOLERANCES)

    # random global transformation
    random = rand_trafo([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # frames for transformed fm
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    frames_prime = call_predictor(fm_prime)
    lorentz_test(frames_prime.matrices, **TOLERANCES)

    # check that frames transform correctly
    # expect frames_prime = frames * random^-1
    inv_random = Frames(random).inv
    frames_prime_expected = torch.einsum("...ij,...jk->...ik", frames.matrices, inv_random)
    torch.testing.assert_close(frames_prime_expected, frames_prime.matrices, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize(*SWEEP)
def test_feature_invariance(
    FramesPredictor, rand_trafo, batch_dims, logm2_std, logm2_mean, predictor_kwargs
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors, **predictor_kwargs).to(dtype=dtype)

    def call_predictor(fm):
        return predictor(fm)

    fm_test = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    predictor.equivectors.init_standardization(fm_test)

    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # random global transformation
    random = rand_trafo([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # path 1: Frames transform (+ random transform)
    frames = call_predictor(fm)
    lorentz_test(frames.matrices, **TOLERANCES)
    fm_local = trafo(fm, frames)

    # path 2: random transform + Frames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    frames_prime = call_predictor(fm_prime)
    lorentz_test(frames_prime.matrices, **TOLERANCES)
    fm_local_prime = trafo(fm_prime, frames_prime)

    # test that features are invariant
    torch.testing.assert_close(fm_local, fm_local_prime, **TOLERANCES)


def test_repr_names_the_orthogonalization_method():
    predictor = LearnedSO13Frames(equivectors=equivectors_builder())
    assert repr(predictor) == "LearnedSO13Frames(method=gramschmidt)"


def test_mass_regularization_lifts_light_particles_onto_the_mass_shell():
    """``mass_reg`` stabilizes the frames by giving near-lightlike particles a minimum mass.

    It is deliberately not part of the sweeps above: rescaling the energy alone is not a
    Lorentz-covariant operation, so it trades equivariance for stability.
    """
    mass_reg = 1.0
    predictor = LearnedSO13Frames(equivectors=equivectors_builder(), mass_reg=mass_reg)
    fm = torch.tensor([[3.0, 1.0, 2.0, 2.0], [10.0, 1.0, 2.0, 2.0]])  # m2 = 0 and m2 = 91

    out = predictor.mass_regularize(fm)
    torch.testing.assert_close(out[:, 1:], fm[:, 1:], **TOLERANCES)  # momenta untouched
    torch.testing.assert_close(out[1], fm[1], **TOLERANCES)  # the heavy particle is left alone
    masses = lorentz_squarednorm(out)
    torch.testing.assert_close(masses[0], torch.tensor(mass_reg**2), **TOLERANCES)


@pytest.mark.parametrize("training", [True, False])
def test_random_frames_resample_only_while_training(training):
    """``random=True`` re-initializes the frames network on every training forward pass, which
    acts as data augmentation; in eval mode the frames have to stay put."""
    predictor = LearnedSO13Frames(equivectors=equivectors_builder(), random=True)
    assert not any(p.requires_grad for p in predictor.equivectors.parameters())
    predictor.train(training)
    fm = sample_particle([10], 1, 0)
    predictor.equivectors.init_standardization(fm)

    first, second = predictor(fm).matrices, predictor(fm).matrices
    if training:
        assert not torch.allclose(first, second)
    else:
        torch.testing.assert_close(first, second, **TOLERANCES)


def test_average_event_averages_within_each_event():
    # three particles carrying the values 1, 2 and 6, in the (item, n_vectors, 4) layout
    vecs = torch.tensor([1.0, 2.0, 6.0]).reshape(3, 1, 1).expand(3, 1, 4)

    # a ptr splits the flat item axis into events: (1, 2) and (6,)
    averaged = average_event(vecs, ptr=torch.tensor([0, 2, 3]))
    torch.testing.assert_close(averaged[:, 0, 0], torch.tensor([1.5, 1.5, 6.0]), **TOLERANCES)

    # without a ptr the batch is dense and the whole particle axis is averaged over
    averaged = average_event(vecs.unsqueeze(0), ptr=None)  # (batch, particle, n_vectors, 4)
    torch.testing.assert_close(averaged[0, :, 0, 0], torch.full((3,), 3.0), **TOLERANCES)
