"""Helpers for using lloca networks with :func:`torch.compile`."""

from collections.abc import Mapping

import torch
from torch import nn


def compile_model(
    model: nn.Module,
    *,
    compile_kwargs: Mapping | None = None,
) -> None:
    """Wrap ``model.forward`` with :func:`torch.compile` in place.

    Rebinding ``self.forward`` rather than patching the class keeps the compilation local
    to this instance.

    Parameters
    ----------
    model
        The :class:`torch.nn.Module` whose ``forward`` should be compiled.
    compile_kwargs
        Forwarded verbatim to :func:`torch.compile` (e.g. ``mode``, ``dynamic``,
        ``fullgraph``, ``backend``). Any key omitted falls back to torch's own default.
    """
    model.forward = torch.compile(model.forward, **dict(compile_kwargs or {}))
