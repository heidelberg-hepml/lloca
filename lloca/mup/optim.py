"""μP-aware optimizers, re-exported from the optional ``mup`` dependency.

These are thin re-exports so that user code can simply do
``from lloca.mup import MuAdamW`` without importing ``mup`` directly, and so the
optional-dependency error message is consistent. They behave exactly like the
standard ``torch`` optimizers for parameters whose width multiplier is 1, so a
model mixing μP and standard-parametrization parameters (see
:func:`lloca.mup.finalize`) can be optimized with a single optimizer.
"""

from .parametrization import _require_mup


def _load():
    _require_mup()
    from mup import MuAdam, MuAdamW, MuReadout, MuSGD

    return MuAdam, MuAdamW, MuSGD, MuReadout


def MuAdam(*args, **kwargs):
    """μP variant of :class:`torch.optim.Adam` (see :func:`mup.MuAdam`)."""
    impl, _, _, _ = _load()
    return impl(*args, **kwargs)


def MuAdamW(*args, **kwargs):
    """μP variant of :class:`torch.optim.AdamW` (see :func:`mup.MuAdamW`)."""
    _, impl, _, _ = _load()
    return impl(*args, **kwargs)


def MuSGD(*args, **kwargs):
    """μP variant of :class:`torch.optim.SGD` (see :func:`mup.MuSGD`)."""
    _, _, impl, _ = _load()
    return impl(*args, **kwargs)
