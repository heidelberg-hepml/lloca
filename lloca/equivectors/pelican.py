"""Edge convolution with PELICAN."""

import math

import torch

from ..utils.lorentz import lorentz_squarednorm
from ..utils.utils import get_edge_attr, scatter
from .base import EquiVectors
from .mlp import get_edge_index_and_batch, get_nonlinearity, get_operation


class PELICANVectors(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        net,
        operation="add",
        nonlinearity="softmax",
        aggr="sum",
        fm_norm=False,
        layer_norm=False,
        use_amp=False,
    ):
        super().__init__()
        self.net = net(in_channels_rank1=num_scalars, out_channels=n_vectors)

        self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

        self.operation = get_operation(operation)
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.aggr = aggr
        self.fm_norm = fm_norm
        self.layer_norm = layer_norm
        self.use_amp = use_amp
        assert not (operation == "single" and fm_norm)  # unstable

    def init_standardization(self, fourmomenta, ptr=None):
        if not self.edge_inited:
            edge_index, _, _ = get_edge_index_and_batch(fourmomenta, ptr)
            fourmomenta = fourmomenta.reshape(-1, 1, 4)
            edge_attr = get_edge_attr(fourmomenta, edge_index)
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited.fill_(True)

    def forward(self, fourmomenta, scalars=None, ptr=None, **kwargs):
        # move to sparse tensors
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])
        edge_index, batch, ptr = get_edge_index_and_batch(fourmomenta, ptr, remove_self_loops=False)
        if len(in_shape) > 1:
            fourmomenta = fourmomenta.reshape(math.prod(in_shape), 4)
            scalars = scalars.reshape(math.prod(in_shape), scalars.shape[-1])

        # compute prefactors
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(scalars.dtype)

        with torch.autocast(fourmomenta.device.type, enabled=self.use_amp):
            prefactor = self.net(
                in_rank2=edge_attr,
                in_rank1=scalars,
                edge_index=edge_index,
                batch=batch,
                num_graphs=ptr.shape[0] - 1,
            )
        row, col = edge_index
        prefactor = self.nonlinearity(
            prefactor, index=row, node_ptr=ptr, node_batch=batch, remove_self_loops=False
        )

        # aggregate relative fourmomenta
        fm_rel = self.operation(fourmomenta[row], fourmomenta[col])
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel = fm_rel / fm_rel_norm.abs().sqrt().clamp(min=1e-6)
        vecs = prefactor.unsqueeze(-1) * fm_rel.unsqueeze(-2)
        vecs = scatter(vecs, row, dim_size=fourmomenta.shape[0], reduce=self.aggr)

        if self.layer_norm:
            norm = lorentz_squarednorm(vecs).sum(dim=-1, keepdim=True).unsqueeze(-1)
            vecs = vecs / norm.abs().sqrt().clamp(min=1e-5)

        # reshape result
        vecs = vecs.reshape(*in_shape, -1, 4)
        return vecs

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)
        return edge_attr
