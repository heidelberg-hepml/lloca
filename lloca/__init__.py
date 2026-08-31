"""LLoCa: Lorentz local canonicalization, making any backbone Lorentz-equivariant."""

from importlib.metadata import version as _pkg_version

from .backbone import LLoCaAttention, LLoCaMessagePassing
from .equivectors import LGATrSlimVectors, LGATrVectors, MLPVectors, PELICANVectors
from .framesnet import Frames, LearnedPDFrames, RandomFrames
from .reps import TensorReps, TensorRepsTransform

__version__ = _pkg_version("lloca")

__all__ = [
    "Frames",
    "LGATrSlimVectors",
    "LGATrVectors",
    "LLoCaAttention",
    "LLoCaMessagePassing",
    "LearnedPDFrames",
    "MLPVectors",
    "PELICANVectors",
    "RandomFrames",
    "TensorReps",
    "TensorRepsTransform",
]
