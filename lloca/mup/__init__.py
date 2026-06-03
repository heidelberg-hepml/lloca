"""Self-contained Maximal-Update Parametrization (μP) support for LLoCa.

Enable μP on any supported backbone by passing ``parametrization="mup"``; the
backbone computes its own base shapes in ``__init__`` (no ``.bsh`` files, no
manual base/delta models). Then mark any parameters living outside μP backbones
with :func:`finalize` and optimize with :class:`MuAdam` / :class:`MuAdamW`::

    from lloca.backbone import Transformer
    import lloca.mup as mup

    net = Transformer(..., parametrization="mup")   # base shapes set automatically
    model = MyModelWrappingNet(net)
    mup.finalize(model)                              # SP-mark external params
    opt = mup.MuAdamW(model.parameters(), lr=lr)

See :mod:`lloca.mup.parametrization` for details and caveats.
"""

from .optim import MuAdam, MuAdamW, MuSGD
from .parametrization import (
    finalize,
    is_mup,
    make_readout,
    mup_available,
    mup_parametrized,
    normal_fanin_,
    reinitialize_mup,
    scale_reps,
)

__all__ = [
    "MuAdam",
    "MuAdamW",
    "MuSGD",
    "finalize",
    "is_mup",
    "make_readout",
    "mup_available",
    "mup_parametrized",
    "normal_fanin_",
    "reinitialize_mup",
    "scale_reps",
]
