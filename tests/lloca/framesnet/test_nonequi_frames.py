import pytest
import torch

from lloca.framesnet.nonequi_frames import (
    IdentityFrames,
    RandomFrames,
)
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_squarednorm
from tests.constants import LOGM2_MEAN_STD, TOLERANCES
from tests.helpers import sample_particle


@pytest.mark.parametrize(
    "FramesPredictor,transform_type",
    [
        (IdentityFrames, None),
        (RandomFrames, "lorentz"),
        (RandomFrames, "rotation"),
        (RandomFrames, "xyrotation"),
    ],
)
@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_vectors(FramesPredictor, transform_type, batch_dims, logm2_mean, logm2_std):
    dtype = torch.float32

    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # predict local frames
    predictor = (
        FramesPredictor(transform_type=transform_type)
        if FramesPredictor == RandomFrames
        else FramesPredictor()
    )
    frames = predictor(fm)

    # transform into local frames
    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))
    fm_local = trafo(fm, frames)

    # every frame is a Lorentz transformation, so the invariant mass is preserved.
    # This is the only check that applies to transform_type="lorentz".
    torch.testing.assert_close(lorentz_squarednorm(fm_local), lorentz_squarednorm(fm), **TOLERANCES)

    if FramesPredictor == IdentityFrames:
        # fourmomenta should not change at all
        torch.testing.assert_close(fm_local, fm, **TOLERANCES)
    elif transform_type == "rotation":
        # a spatial rotation leaves the energy unchanged
        torch.testing.assert_close(fm_local[..., [0]], fm[..., [0]], **TOLERANCES)
    elif transform_type == "xyrotation":
        # a rotation in the xy-plane leaves energy and pz unchanged
        torch.testing.assert_close(fm_local[..., [0, 3]], fm[..., [0, 3]], **TOLERANCES)
