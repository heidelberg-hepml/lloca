"""Edge convolution with L-GATr."""

import math

import torch
from lgatr import embed_vector
from lgatr.layers import EquiLayerNorm
from lgatr.nets.lgatr_slim import RMSNorm
from lgatr.primitives.invariants import _load_inner_product_factors
from torch_geometric.nn import MessagePassing

from ..backbone.attention_backends.mask import get_sparse_attention_mask
from ..utils.lorentz import lorentz_squarednorm
from ..utils.utils import get_batch_from_ptr
from .base import EquiVectors
from .mlp import get_edge_index_and_batch, get_nonlinearity, get_operation


class _LGATrVectorsBase(EquiVectors, MessagePassing):
    """Shared machinery for L-GATr-based equivariant vector predictors.

    Subclasses set up ``self.net`` and ``self.lgatr_norm`` and implement
    ``_embed_input`` (how four-momenta enter the network) and ``_get_qk_metric``
    (the metric contracting the query/key vector channels).
    """

    def __init__(
        self,
        n_vectors,
        operation="add",
        nonlinearity="softmax",
        aggr="sum",
        layer_norm=False,
        use_amp=False,
        attention_backend="xformers",
    ):
        # Note: fm_norm option not supported, because it would be unstable with remove_self_loops=False
        super().__init__(aggr=aggr)
        self.n_vectors = n_vectors
        self.operation = get_operation(operation)
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.layer_norm = layer_norm
        self.use_amp = use_amp
        self.attention_backend = attention_backend

    @staticmethod
    def _doubled_channels(hidden_channels, n_vectors):
        if hidden_channels <= 0:
            return 0
        return 2 * n_vectors * max(1, hidden_channels // (2 * n_vectors))

    def _embed_input(self, fourmomenta):
        raise NotImplementedError

    def _get_qk_metric(self, device, dtype):
        raise NotImplementedError

    def forward(self, fourmomenta, scalars=None, ptr=None, **kwargs):
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])

        attn_kwargs = {}
        if ptr is not None:
            batch = get_batch_from_ptr(ptr)
            attn_kwargs = get_sparse_attention_mask(
                batch, attention_backend=self.attention_backend, dtype=scalars.dtype
            )
        edge_index, batch, ptr = get_edge_index_and_batch(fourmomenta, ptr, remove_self_loops=False)

        fourmomenta = fourmomenta.unsqueeze(0)
        scalars = scalars.unsqueeze(0)

        # get query and key from the underlying L-GATr network
        net_input = self._embed_input(fourmomenta).to(scalars.dtype)
        with torch.autocast(net_input.device.type, enabled=self.use_amp):
            qk_v, qk_s = self.net(net_input, scalars, **attn_kwargs)
        if self.lgatr_norm is not None:
            qk_v, qk_s = self.lgatr_norm(qk_v, qk_s)

        # flatten for message passing
        fm_shape = fourmomenta.shape[:-1]
        fourmomenta = fourmomenta.reshape(math.prod(fm_shape), 4)
        qk_v = qk_v.reshape(math.prod(fm_shape), qk_v.shape[-2], qk_v.shape[-1])
        qk_s = qk_s.reshape(math.prod(fm_shape), qk_s.shape[-1])

        # extract q and k
        q_v, k_v = torch.chunk(qk_v.to(fourmomenta.dtype), chunks=2, dim=-2)
        q_s, k_s = torch.chunk(qk_s.to(fourmomenta.dtype), chunks=2, dim=-1)

        # unpack the n_vectors axis
        q_v = q_v.reshape(*q_v.shape[:-2], self.n_vectors, -1, q_v.shape[-1])
        k_v = k_v.reshape(*k_v.shape[:-2], self.n_vectors, -1, k_v.shape[-1])
        q_s = q_s.reshape(*q_s.shape[:-1], self.n_vectors, -1)
        k_s = k_s.reshape(*k_s.shape[:-1], self.n_vectors, -1)

        qk_product = self._get_qk_product(q_v, k_v, q_s, k_s, edge_index)

        # message-passing
        vecs = self.propagate(
            edge_index,
            fm=fourmomenta,
            prefactor=qk_product,
            batch=batch,
            node_ptr=ptr,
        )
        vecs = vecs.reshape(fourmomenta.shape[0], -1, 4)

        if self.layer_norm:
            norm = lorentz_squarednorm(vecs).sum(dim=-1, keepdim=True).unsqueeze(-1)
            vecs = vecs / norm.abs().sqrt().clamp(min=1e-5)

        # reshape result
        vecs = vecs.reshape(*in_shape, -1, 4)
        return vecs

    def message(
        self,
        edge_index,
        fm_i,
        fm_j,
        node_ptr,
        batch,
        prefactor,
    ):
        # prepare fourmomenta
        fm_rel = self.operation(fm_i, fm_j)
        fm_rel = fm_rel[:, None, :4]

        prefactor = self.nonlinearity(
            prefactor,
            index=edge_index[0],
            node_ptr=node_ptr,
            node_batch=batch,
            remove_self_loops=False,
        )
        prefactor = prefactor.unsqueeze(-1)
        out = prefactor * fm_rel
        out = out.reshape(out.shape[0], -1)
        return out

    def _get_qk_product(self, q_v, k_v, q_s, k_s, edge_index):
        metric = self._get_qk_metric(device=q_v.device, dtype=q_v.dtype)
        q = torch.cat([(q_v * metric).flatten(-2, -1), q_s], dim=-1)
        k = torch.cat([k_v.flatten(-2, -1), k_s], dim=-1)

        # evaluate attention weights on edges
        scale_factor = 1 / math.sqrt(q.shape[-1])
        src, dst = edge_index
        q_edges, k_edges = q[src], k[dst]
        qk_product = (q_edges * k_edges).sum(dim=-1) * scale_factor
        return qk_product


class LGATrVectors(_LGATrVectorsBase):
    """Wrapper around the multivector ``lgatr.nets.LGATr`` backbone."""

    def __init__(
        self,
        n_vectors,
        num_scalars,
        hidden_mv_channels,
        hidden_s_channels,
        net,
        operation="add",
        nonlinearity="softmax",
        aggr="sum",
        layer_norm=False,
        lgatr_norm=True,
        use_amp=False,
        attention_backend="xformers",
    ):
        super().__init__(
            n_vectors=n_vectors,
            operation=operation,
            nonlinearity=nonlinearity,
            aggr=aggr,
            layer_norm=layer_norm,
            use_amp=use_amp,
            attention_backend=attention_backend,
        )
        out_mv_channels = self._doubled_channels(hidden_mv_channels, n_vectors)
        out_s_channels = self._doubled_channels(hidden_s_channels, n_vectors)
        self.net = net(
            in_s_channels=num_scalars,
            out_mv_channels=out_mv_channels,
            out_s_channels=out_s_channels,
        )
        self.lgatr_norm = EquiLayerNorm() if lgatr_norm else None

    def _embed_input(self, fourmomenta):
        return embed_vector(fourmomenta).unsqueeze(-2)

    def _get_qk_metric(self, device, dtype):
        return _load_inner_product_factors(device=device, dtype=dtype)


class LGATrSlimVectors(_LGATrVectorsBase):
    """Wrapper around the vector-stream ``lgatr.nets.LGATrSlim`` backbone."""

    def __init__(
        self,
        n_vectors,
        num_scalars,
        hidden_v_channels,
        hidden_s_channels,
        net,
        operation="add",
        nonlinearity="softmax",
        aggr="sum",
        layer_norm=False,
        lgatr_norm=True,
        use_amp=False,
        attention_backend="xformers",
    ):
        super().__init__(
            n_vectors=n_vectors,
            operation=operation,
            nonlinearity=nonlinearity,
            aggr=aggr,
            layer_norm=layer_norm,
            use_amp=use_amp,
            attention_backend=attention_backend,
        )
        out_v_channels = self._doubled_channels(hidden_v_channels, n_vectors)
        out_s_channels = self._doubled_channels(hidden_s_channels, n_vectors)
        self.net = net(
            in_s_channels=num_scalars,
            out_v_channels=out_v_channels,
            out_s_channels=out_s_channels,
        )
        self.lgatr_norm = (
            RMSNorm(out_v_channels, out_s_channels, elementwise_affine=False)
            if lgatr_norm
            else None
        )

    def _embed_input(self, fourmomenta):
        return fourmomenta.unsqueeze(-2)

    def _get_qk_metric(self, device, dtype):
        # Minkowski metric, contracting the four components of each vector channel
        return torch.tensor([1.0, -1.0, -1.0, -1.0], device=device, dtype=dtype)
