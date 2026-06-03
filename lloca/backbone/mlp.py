"""Simple MLP module."""

import math

import torch
from torch import nn

from ..mup import make_readout, mup_parametrized, reinitialize_mup


@mup_parametrized
class MLP(nn.Module):
    """A simple MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.

    Parameters
    ----------
    in_shape, out_shape, hidden_channels, hidden_layers, dropout_prob
        Standard MLP arguments.
    parametrization : str
        ``"sp"`` (standard, default) or ``"mup"``. Under μP the width axis is
        ``hidden_channels``: the final layer becomes a :class:`mup.MuReadout`, the
        weights are μP-initialized, and the base shapes are computed automatically.
        See :mod:`lloca.mup`.
    mup_base_shapes, mup_delta_shapes : dict, optional
        Base/delta width overrides; default ``{"hidden_channels": 64}`` /
        ``{"hidden_channels": 128}``.
    """

    # Default base/delta width overrides for μP base-shape computation.
    DEFAULT_MUP_SHAPES = ({"hidden_channels": 64}, {"hidden_channels": 128})

    def __init__(
        self,
        in_shape,
        out_shape,
        hidden_channels,
        hidden_layers,
        dropout_prob=None,
        *,
        parametrization: str = "sp",
        mup_base_shapes: dict | None = None,
        mup_delta_shapes: dict | None = None,
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.parametrization = parametrization
        self.in_shape = in_shape
        self.out_shape = out_shape

        layers: list[nn.Module] = [nn.Linear(prod(in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.GELU())
        layers.append(make_readout(hidden_channels, prod(self.out_shape), parametrization))
        self.mlp = nn.Sequential(*layers)

        if parametrization == "mup":
            reinitialize_mup(self)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of MLP."""
        return self.mlp(inputs)


def prod(shape):
    if isinstance(shape, int):
        return shape
    else:
        return math.prod(shape)
