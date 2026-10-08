"""Edge convolution with a simple MLP."""

import torch

from ..backbone.mlp import MLP
from ..utils.lorentz import lorentz_squarednorm
from ..utils.utils import (
    get_batch_from_ptr,
    get_edge_attr,
    get_edge_index_from_ptr,
    get_edge_index_from_shape,
    get_node_to_edge_ptr_fully_connected,
    scatter,
)
from .base import EquiVectors


class MLPVectors(EquiVectors):
    """Edge convolution with a simple MLP."""

    def __init__(
        self,
        n_vectors,
        num_scalars,
        hidden_channels,
        num_layers_mlp,
        include_edges=True,
        operation="add",
        nonlinearity="softmax",
        fm_norm=True,
        layer_norm=True,
        use_amp=False,
        dropout_prob=None,
        aggr="sum",
    ):
        """Equivariant edge convolution on a fully connected graph.

        The choice of the parameters ``operation``, ``nonlinearity``, ``fm_norm``, ``aggr``, ``layer_norm`` is critical to the stability of the approach.
        Bad combinations initialize the ``framesnet`` to predict strongly boosted vectors, leading to strongly boosted frames and unstable training.
        We recommend to stick to the default parameters, which worked for all our experiments.

        Parameters
        ----------
        n_vectors : int
            Number of output vectors per particle.
            Different FramesPredictor's need different n_vectors,
            so this parameter should be set dynamically.
        num_scalars : int
            Number of scalar features per particle.
        hidden_channels : int
            Number of hidden channels in the MLP.
        num_layers_mlp : int
            Number of hidden layers in the MLP.
        include_edges : bool
            Whether to include edge attributes in the message passing. If True, edge attributes will be calculated from fourmomenta and standardized. Default is True.
        operation : str
            Operation to perform on the fourmomenta. Options are "add", "diff", or "single". Default is "add".
        nonlinearity : str
            Nonlinearity to apply to the output of the MLP. Options are "exp", "softplus", and "softmax". Default is "softmax".
        fm_norm : bool
            Whether to normalize the relative fourmomentum. Default is True.
        layer_norm : bool
            Whether to apply Lorentz-equivariant layer normalization to the output vectors. Default is True.
        use_amp : bool
            Whether to use automatic mixed precision (AMP) for the MLP. Default is False.
        dropout_prob : float
            Dropout probability for the MLP. If None, no dropout will be applied. Default is None.
        aggr : str
            Aggregation method for message passing. Options are "sum", "mean", or "max". Default is "sum".
        """
        super().__init__()
        assert num_scalars > 0 or include_edges, (
            "Either num_scalars > 0 or include_edges==True, otherwise there are no inputs."
        )
        self.include_edges = include_edges
        self.layer_norm = layer_norm
        self.operation = get_operation(operation)
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.fm_norm = fm_norm
        assert not (operation == "single" and fm_norm), (
            "The setup operation=single and fm_norm==True is unstable"
        )
        self.use_amp = use_amp
        self.aggr = aggr

        in_channels = 2 * num_scalars + int(include_edges)
        self.mlp = MLP(
            in_shape=[in_channels],
            out_shape=n_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

        if include_edges:
            self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))
            self._edge_inited_checked = False

    def init_standardization(self, fourmomenta, ptr=None):
        if self.include_edges and not self.edge_inited:
            edge_index, _, _ = get_edge_index_and_batch(fourmomenta, ptr)
            edge_attr = get_edge_attr(fourmomenta.reshape(-1, 4), edge_index)
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited.fill_(True)
            self._edge_inited_checked = True

    def forward(self, fourmomenta, scalars=None, ptr=None, **kwargs):
        """
        Parameters
        ----------
        fourmomenta : torch.Tensor
            Tensor of shape (..., 4) containing the fourmomenta of the particles.
        scalars : torch.Tensor, optional
            Tensor of shape (..., num_scalars) containing scalar features for each particle. If None, a tensor of zeros will be created.
        ptr : torch.Tensor, optional
            Pointer tensor indicating the start and end of each batch for sparse tensors.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., n_vectors, 4) containing the predicted vectors for each particle.
        """
        # move to sparse tensors
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])
        edge_index, batch, ptr = get_edge_index_and_batch(fourmomenta, ptr)
        fourmomenta = fourmomenta.reshape(-1, 4)
        scalars = scalars.reshape(fourmomenta.shape[0], scalars.shape[-1])
        row, col = edge_index

        # MLP on the edges
        prefactor = torch.cat([scalars[row], scalars[col]], dim=-1)
        if self.include_edges:
            if not self._edge_inited_checked:
                assert self.edge_inited
                self._edge_inited_checked = True
            edge_attr = get_edge_attr(fourmomenta, edge_index)
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std

            # fourmomenta may be float64
            edge_attr = edge_attr.to(scalars.dtype)
            prefactor = torch.cat([prefactor, edge_attr.unsqueeze(-1)], dim=-1)
        with torch.autocast(prefactor.device.type, enabled=self.use_amp):
            prefactor = self.mlp(prefactor)
        prefactor = self.nonlinearity(prefactor, index=row, node_ptr=ptr, node_batch=batch)

        # aggregate relative fourmomenta
        fm_rel = self.operation(fourmomenta[row], fourmomenta[col])
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel = fm_rel / fm_rel_norm.abs().sqrt().clamp(min=1e-6)
        vecs = prefactor.unsqueeze(-1) * fm_rel.unsqueeze(-2)
        vecs = scatter(vecs, row, dim_size=fourmomenta.shape[0], reduce=self.aggr)

        # equivariant layer normalization
        if self.layer_norm:
            norm = lorentz_squarednorm(vecs).sum(dim=-1, keepdim=True).unsqueeze(-1)
            vecs = vecs / norm.abs().sqrt().clamp(min=1e-5)
        return vecs.reshape(*in_shape, -1, 4)


def softmax(src, ptr):
    r"""Adapted version of the torch_geometric softmax function
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/_softmax.html.
    Pass output_size to torch.repeat_interleave to avoid GPU/CPU sync.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor of shape (N, ...) where N is the number of elements.
        The softmax is applied along the first dimension.
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.
        Tensor of shape (B+1,) where B is the number of batches.
    """
    count = ptr[1:] - ptr[:-1]
    src_max = torch._segment_reduce(src.detach(), "amax", offsets=ptr)
    src_max = src_max.repeat_interleave(count, dim=0, output_size=src.shape[0])
    out = (src - src_max).exp()
    out_sum = torch._segment_reduce(out, "sum", offsets=ptr) + 1e-16
    out_sum = out_sum.repeat_interleave(count, dim=0, output_size=src.shape[0])
    return out / out_sum


def _single(fm_i, fm_j):
    return fm_j


def _exp(x, *args, **kwargs):
    return torch.clamp(x, min=-10, max=10).exp()


def _softplus(x, *args, **kwargs):
    return torch.nn.functional.softplus(x)


def _softmax_fully_connected(x, index, node_ptr, node_batch, remove_self_loops=True):
    edge_ptr = get_node_to_edge_ptr_fully_connected(
        node_ptr, node_batch, remove_self_loops=remove_self_loops
    )
    return softmax(x, ptr=edge_ptr)


def get_operation(operation):
    """
    Parameters
    ----------
    operation : str
        Operation to perform on the fourmomenta. Options are "add", "diff", or "single".

    Returns
    -------
    callable
        A function that performs the specified operation on two fourmomenta tensors.
    """
    if operation == "diff":
        return torch.sub
    elif operation == "add":
        return torch.add
    elif operation == "single":
        return _single
    else:
        raise ValueError(f"Invalid operation {operation}. Options are (add, diff, single).")


def get_nonlinearity(nonlinearity):
    """
    Parameters
    ----------
    nonlinearity : str
        Nonlinearity to apply to the output of the MLP. Options are "exp", "softplus", "softmax".
        We enforce the prediction of timelike vectors.

    Returns
    -------
    callable
        A function that applies the specified nonlinearity to the input tensor.
    """
    if nonlinearity == "exp":
        return _exp
    elif nonlinearity == "softplus":
        return _softplus
    elif nonlinearity == "softmax":
        return _softmax_fully_connected
    else:
        raise ValueError(
            f"Invalid nonlinearity {nonlinearity}. Options are (exp, softplus, softmax)."
        )


def get_edge_index_and_batch(fourmomenta, ptr, remove_self_loops=True):
    if ptr is not None:
        assert fourmomenta.dim() == 2, "ptr only supported for sparse tensors"
        edge_index, batch = _get_edge_index_and_batch_from_ptr(
            ptr, fourmomenta.shape, remove_self_loops
        )
    else:
        shape = fourmomenta.shape if fourmomenta.dim() > 2 else (1, *fourmomenta.shape)
        edge_index, batch = get_edge_index_from_shape(
            shape, fourmomenta.device, remove_self_loops=remove_self_loops
        )
        ptr = torch.arange(shape[0] + 1, device=fourmomenta.device) * shape[1]
    return edge_index, batch, ptr


@torch.compiler.disable
def _get_edge_index_and_batch_from_ptr(ptr, shape, remove_self_loops):
    # inductor is slow or fails on the data-dependent number of edges
    edge_index = get_edge_index_from_ptr(ptr, shape=shape, remove_self_loops=remove_self_loops)
    batch = get_batch_from_ptr(ptr, num_items=shape[0])
    return edge_index, batch
