"""Transforming tensors of different Lorentz group representations."""

import torch

from ..framesnet.frames import Frames
from ..utils.autocast import minimum_autocast_precision
from .tensorreps import TensorReps


class TensorRepsTransform(torch.nn.Module):
    def __init__(
        self,
        reps: TensorReps,
    ):
        """Tensor representation transformation module.

        Parameters
        ----------
        reps: TensorReps
            Tensor representations to transform, sorted by order.
        """
        super().__init__()
        assert reps.is_sorted, f"reps have to be sorted by order, but got {reps}"
        self.reps = reps
        self.max_order = reps.max_rep.rep.order

        # dimension covered by each order; because the reps are sorted, the reps of a given
        # order (there can be two of them, one per parity) form a single contiguous block
        dims = [0] * (self.max_order + 1)
        parity_odd = torch.zeros(reps.dim, dtype=torch.bool)
        idx = 0
        for mul_rep in reps:
            dims[mul_rep.rep.order] += mul_rep.dim
            parity_odd[idx : idx + mul_rep.dim] = mul_rep.rep.parity == -1
            idx += mul_rep.dim

        # start and end index of the block of each order
        self.blocks, start = [], 0
        for dim in dims:
            self.blocks.append((start, start + dim))
            start += dim
        self.dim_scalars = dims[0]

        # parity-odd states pick up a factor sign(det Lambda)
        self.register_buffer("parity_odd", parity_odd.unsqueeze(0))
        self.no_parity_odd = not parity_odd.any().item()

    @minimum_autocast_precision(torch.float32)
    def forward(self, tensor: torch.Tensor, frames: Frames):
        """Apply a transformation to a tensor of a given representation.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to transform, shape (..., self.reps.dim).
        frames: Frames
            The local frames to apply the transformation with, shape (..., 4, 4).

        Returns
        -------
        torch.Tensor
            The transformed tensor, shape (..., self.reps.dim).
        """
        if frames.is_identity or (self.no_parity_odd and self.max_order == 0):
            return tensor

        # flatten the batch dimensions, everything below operates on a single batch dimension
        shape = tensor.shape
        tensor = tensor.reshape(-1, shape[-1])
        matrices = frames.matrices.reshape(-1, 4, 4)
        assert tensor.shape[0] == matrices.shape[0], (
            f"Batch dimension is {tensor.shape[0]} for tensor, but {matrices.shape[0]} for frames."
        )

        if self.max_order > 0:
            tensor = self._transform(tensor, matrices.to(tensor.dtype))
        if not self.no_parity_odd:
            tensor = self._transform_parity(tensor, frames.det.reshape(-1, 1))
        return tensor.view(shape)

    def _transform(self, tensor, matrices):
        # Transform one tensor index at a time, starting at the highest order. Each transformed
        # index is absorbed into the channel dimension C, so one operation per index handles all
        # reps that still have untransformed indices left.
        out = None
        for order in range(self.max_order, 0, -1):
            start, end = self.blocks[order]
            if end > start:
                block = tensor[:, start:end].unflatten(-1, (-1, 4**order))
                out = block if out is None else torch.cat((block, out), dim=-2)

            # contract the leading index of (N, C, 4, R) with the frames and merge it into C
            out = out.unflatten(-1, (4, 4 ** (order - 1)))
            out = (matrices[:, None, :, :, None] * out[:, :, None]).sum(-2).flatten(-3, -2)

        out = out.flatten(-2, -1)
        if self.dim_scalars == 0:
            return out
        return torch.cat((tensor[:, : self.dim_scalars], out), dim=-1)

    def _transform_parity(self, tensor, det):
        # Parity transform: multiply parity-odd states by sign(det Lambda)
        return torch.where(self.parity_odd, det.sign().to(tensor.dtype) * tensor, tensor)
