"""Baseline LLoCa-Transformer."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..mup import make_readout, mup_parametrized, normal_fanin_
from ..reps.tensorreps import TensorReps
from .attention import LLoCaAttention


class BaselineLayerNorm(nn.Module):
    """Baseline layer norm over all dimensions except the first."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        outputs : Tensor
            Normalized inputs.
        """
        return torch.nn.functional.layer_norm(inputs, normalized_shape=inputs.shape[-1:])


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_channels, 3 * hidden_channels)

    def mup_reset_parameters(self):
        """μP init: fan-in normal weights, then zero the query block.

        ``self.linear`` produces ``[q | k | v]`` along its output features (the
        leading ``3`` axis of the ``(..., 3, hidden, heads)`` view), so the queries
        are the first third of the output rows. Zeroing them makes the initial
        attention logits vanish (uniform attention at init), which stabilizes μP
        width transfer.
        """
        normal_fanin_(self.linear)
        n_q = self.linear.out_features // 3
        nn.init.zeros_(self.linear.weight[:n_q])

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels)

        *leading, items, last = qkv.shape
        hidden_channels = last // (3 * self.num_heads)
        qkv = qkv.view(*leading, items, 3, hidden_channels, self.num_heads)
        qkv = qkv.movedim(-3, 0).movedim(-1, len(leading) + 1)
        q, k, v = qkv.unbind(dim=0)  # 3x (..., num_heads, num_items, hidden_channels // num_heads)
        return q, k, v


class MultiQueryQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-query attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_channels, hidden_channels)
        self.k_linear = nn.Linear(in_channels, hidden_channels // num_heads)
        self.v_linear = nn.Linear(in_channels, hidden_channels // num_heads)

    def mup_reset_parameters(self):
        """μP init: fan-in normal keys/values, zero queries (see MultiHeadQKVLinear)."""
        normal_fanin_(self.k_linear)
        normal_fanin_(self.v_linear)
        nn.init.zeros_(self.q_linear.weight)
        if self.q_linear.bias is not None:
            nn.init.zeros_(self.q_linear.bias)

    def forward(self, inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        q = self.q_linear(inputs)

        *leading, items, hidden_channels = q.shape
        q = q.reshape(*leading, items, self.num_heads, hidden_channels // self.num_heads)
        q = q.movedim(-2, -3)

        k = self.k_linear(inputs)[
            ..., None, :, :
        ]  # (..., head=1, item, hidden_channels // num_heads)
        v = self.v_linear(inputs)[..., None, :, :]
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    attention
    num_heads : int
        Number of attention heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    dropout_prob : float
        Dropout probability for output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        attention,
        num_heads: int = 8,
        multi_query: bool = True,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.attention = attention

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels, out_channels)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(inputs)  # each: (..., num_heads, num_items, num_channels)

        # Attention layer
        h = self.attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q),
            **attn_kwargs,
        )

        # Concatenate heads and transform linearly
        *leading, num_heads, num_items, channels_per_head = h.shape
        h = h.permute(*range(len(leading)), -2, -3, -1)
        h = h.reshape(*leading, num_items, num_heads * channels_per_head)

        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another LayerNorm, an
    item-wise two-layer MLP with GeLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    attention
    num_heads : int
        Number of attention heads.
    attention_factor : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    mlp_factor : int
        Factor by which the activation size is increased over the default value of hidden_channels.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    dropout_prob : float
        Dropout probability for output.
    """

    def __init__(
        self,
        hidden_channels,
        attention,
        num_heads: int = 8,
        attention_factor: int = 1,
        multi_query: bool = True,
        mlp_factor: int = 4,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        self.norm1 = BaselineLayerNorm()
        self.norm2 = BaselineLayerNorm()

        hidden_channels_attn = hidden_channels * attention_factor

        self.attention = BaselineSelfAttention(
            hidden_channels,
            hidden_channels,
            hidden_channels_attn,
            attention,
            num_heads=num_heads,
            multi_query=multi_query,
            dropout_prob=dropout_prob,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_factor * hidden_channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
            nn.GELU(),
            nn.Linear(mlp_factor * hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm1(inputs)
        h = self.attention(h, **attn_kwargs)
        outputs = inputs + h

        # Residual MLP
        h = self.norm2(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs


@mup_parametrized
class Transformer(nn.Module):
    """Baseline LLoCa-Transformer.

    Combines transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    attn_reps : str
        Representation of each attention head.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    checkpoint_blocks : bool
        Use gradient checkpointing for transformer blocks.
    attention_factor : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    mlp_factor : int
        Factor by which the activation size is increased over the default value of hidden_channels.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    dropout_prob : float
        Dropout probability for output.
    compile : bool, optional
        Whether to compile the model with torch.compile, by default False.
    compile_mode : str
        torch.compile compilation mode, see torch docs for more information.
    compile_dynamic : bool
        Whether to use dynamic shapes with torch.compile, by default True.
    parametrization : str
        ``"sp"`` (standard, default) or ``"mup"``. Under μP the width axis is
        ``num_heads`` (``hidden_channels = attn_reps.dim * num_heads``); the readout
        becomes a :class:`mup.MuReadout`, the attention is scaled by ``1/d``, the
        weights are μP-initialized, and the base shapes are computed automatically
        from the base/delta widths -- no ``.bsh`` file or manual base/delta models.
        See :mod:`lloca.mup`.
    mup_base_shapes, mup_delta_shapes : dict, optional
        Constructor-argument overrides defining the base/delta widths used to compute
        the μP base shapes. Default to ``{"num_heads": 2}`` / ``{"num_heads": 4}``.
    """

    # Default base/delta width overrides for μP base-shape computation.
    DEFAULT_MUP_SHAPES = ({"num_heads": 2}, {"num_heads": 4})

    def __init__(
        self,
        in_channels: int,
        attn_reps: str,
        out_channels: int,
        num_blocks: int,
        num_heads: int,
        checkpoint_blocks: bool = False,
        attention_factor: int = 1,
        mlp_factor: int = 4,
        multi_query: bool = False,
        dropout_prob: float | None = None,
        compile: bool = False,
        compile_mode: str = "default",
        compile_dynamic: bool = True,
        *,
        parametrization: str = "sp",
        mup_base_shapes: dict | None = None,
        mup_delta_shapes: dict | None = None,
    ) -> None:
        super().__init__()
        self.parametrization = parametrization
        attn_reps = TensorReps(attn_reps)
        self.hidden_channels = attn_reps.dim * num_heads // attention_factor
        self.checkpoint_blocks = checkpoint_blocks
        self.attention = LLoCaAttention(attn_reps, num_heads, parametrization=parametrization)

        self.linear_in = nn.Linear(in_channels, self.hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    self.hidden_channels,
                    attention=self.attention,
                    num_heads=num_heads,
                    attention_factor=attention_factor,
                    mlp_factor=mlp_factor,
                    multi_query=multi_query,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = make_readout(self.hidden_channels, out_channels, parametrization)

        if parametrization == "mup":
            self._mup_reset_parameters()

        if compile:
            # ugly hack to make torch.compile convenient for users
            # the clean solution is model = torch.compile(model, **kwargs) outside of the constructor
            # note that we need fullgraph=False because of the torch.compiler.disable for attention
            self.__class__ = torch.compile(
                self.__class__, dynamic=compile_dynamic, mode=compile_mode
            )

    def _mup_reset_parameters(self) -> None:
        """μP base initialization (applied before ``set_base_shapes`` rescales).

        Hidden/input/value/key linears get a fan-in normal; queries and the readout
        are zero-initialized. ``mup.set_base_shapes(rescale_params=True)`` then
        applies the width-dependent rescaling on top of this width-independent base.
        """
        normal_fanin_(self.linear_in)
        for block in self.blocks:
            block.attention.qkv_linear.mup_reset_parameters()
            normal_fanin_(block.attention.out_linear)
            for layer in block.mlp:
                if isinstance(layer, nn.Linear):
                    normal_fanin_(layer)
        # linear_out is a MuReadout (readout_zero_init=True already zeros the weight)
        nn.init.zeros_(self.linear_out.weight)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, inputs: torch.Tensor, frames, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_items, in_channels)
        frames : Frames
            Local frames used for invariant particle attention
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs with shape (..., num_items, out_channels)
        """
        self.attention.prepare_frames(frames)

        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                h = checkpoint(block, h, use_reentrant=False, **attn_kwargs)
            else:
                h = block(h, **attn_kwargs)
        outputs = self.linear_out(h)
        return outputs
