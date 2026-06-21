"""Helpers for using lloca networks with :func:`torch.compile`."""

import torch
from torch import nn


def compile_model(
    model: nn.Module,
    *,
    compile_mode: str = "default",
    compile_dynamic: bool = False,
    compile_fullgraph: bool = False,
) -> None:
    """Wrap ``model.forward`` with :func:`torch.compile` in place.

    Rebinding ``self.forward`` rather than patching the class keeps the compilation local
    to this instance.

    Parameters
    ----------
    model
        The :class:`torch.nn.Module` whose ``forward`` should be compiled.
    compile_mode
        Mode passed to :func:`torch.compile` (e.g. ``"default"``, ``"reduce-overhead"``).
    compile_dynamic
        Whether to use dynamic shapes.
    compile_fullgraph
        Whether to require a full graph (no graph breaks). Kept ``False`` here because the
        attention path uses ``torch.compiler.disable``.
    """
    model.forward = torch.compile(
        model.forward,
        mode=compile_mode,
        dynamic=compile_dynamic,
        fullgraph=compile_fullgraph,
    )
