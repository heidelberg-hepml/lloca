"""LLoCa: Lorentz local canonicalization, making any backbone Lorentz-equivariant."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .backbone import LLoCaAttention, LLoCaMessagePassing
from .equivectors import LGATrSlimVectors, LGATrVectors, MLPVectors, PELICANVectors
from .framesnet import Frames, LearnedPDFrames, RandomFrames
from .reps import TensorReps, TensorRepsTransform

# Version comes from setuptools-scm via the installed package; falls back for uninstalled checkouts.
try:
    __version__ = _pkg_version("lloca")
except PackageNotFoundError:
    __version__ = "0.0.0"

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
