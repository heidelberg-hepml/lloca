"""LLoCa attention module."""

from math import prod

import torch
from torch import Tensor

from ..framesnet.frames import Frames, InverseFrames, LowerIndicesFrames
from ..reps.tensorreps import TensorReps
from ..reps.tensorreps_transform import TensorRepsTransform
from ..utils.autocast import minimum_autocast_precision
from ..utils.lorentz import lorentz_squarednorm
from ..utils.utils import get_batch_from_ptr
from .attention_backends import get_attention_backend


def _scale_frames(frames: Frames, scale: torch.Tensor) -> Frames:
    """Uniformly rescale each frame matrix by a per-particle scalar factor.

    A grade-n tensor transform applies the frame matrix n times, so scaling the matrices by
    ``scale`` rescales grade-n channels by ``scale**n`` (grade 0 / scalars are untouched) --
    the same effect as dividing the post-transform tensor by a per-channel gamma**grade
    divisor, but folded once into the (much smaller) frame matrices instead of applied to
    every q/k/v/output tensor in every layer.

    Parameters
    ----------
    frames: Frames
        Frames to rescale.
    scale: torch.Tensor
        Per-particle scalar factor, shape (..., 1, 1) broadcastable against
        ``frames.matrices``.

    Returns
    -------
    Frames
    """
    return Frames(
        matrices=frames.matrices * scale,
        is_global=frames.is_global,
        inv=frames.inv / scale,
        det=frames.det * scale[..., 0, 0] ** 4,
        parity=frames.parity,  # scale > 0, so the parity is unchanged
    )


class LLoCaAttention(torch.nn.Module):
    def __init__(
        self,
        attn_reps,
        num_heads,
        preserve_variance=True,
        variance_eps=1e-2,
    ):
        """Attention with frame-to-frame transformations.

        Parameters
        ----------
        attn_reps : TensorReps
            Tensor representation of a single attention head.
        num_heads : int
            Number of attention heads
        preserve_variance : bool
            Rescale the pre-attention (local->global) q/k/v and post-attention (global->local) vectors
            by 1/gamma_i^grade to prevent the variance blowup from large boosts. Needs the reference
            momentum ``p_ref`` in :meth:`prepare_frames`.
        variance_eps : float
            Small mass floor (energy units) that keeps gamma_i finite for near-lightlike jets.
        """
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))
        self.num_heads = num_heads
        self.preserve_variance = preserve_variance
        self.variance_eps = variance_eps

        self.frames = None
        self.frames_qkv = None
        self.frames_out = None

    def _compute_gamma(self, frames, p_ref, ptr=None):
        """Invariant per-particle Lorentz factor gamma_i >= 1 that prevents variance blowup."""
        dtype = torch.promote_types(p_ref.dtype, torch.float32)
        L = frames.matrices.to(dtype)
        p_ref = p_ref.to(dtype)
        if ptr is None:
            # dense: one reference momentum per event, broadcast over the token axis
            p_ref = p_ref.unsqueeze(-2).expand(*L.shape[:-2], 4)
        else:
            # packed: map the per-jet reference momentum to each token
            seg = get_batch_from_ptr(ptr, num_items=L.shape[-3])
            p_ref = p_ref.index_select(0, seg)
        m_ref = torch.sqrt(self.variance_eps**2 + lorentz_squarednorm(p_ref).clamp(min=0))
        gamma = torch.einsum("...nij,...nj->...ni", L, p_ref)[..., 0] / m_ref
        return gamma.detach()  # fixed normalization: no gradient into the frames

    @minimum_autocast_precision(torch.float32)
    def prepare_frames(self, frames, p_ref=None, ptr=None):
        """Prepare local frames for LLoCa attention (called once per forward pass).

        Parameters
        ----------
        frames: Frames
            Local frames of shape (..., N, 4, 4).
        p_ref: torch.tensor, optional
            Reference 4-momentum in the global frame (energy-first), i.e. the total (jet) momentum:
            per event ``(..., 4)`` for a dense layout, or per jet ``(num_jets, 4)`` with ``ptr`` for
            a packed layout. Required when the ``preserve_variance`` flag is on, ignored otherwise.
        ptr: torch.tensor, optional
            Jet boundaries for a packed layout; maps the per-jet ``p_ref`` to each token.
        """
        if len(frames.shape) < 3 or tuple(frames.shape[-2:]) != (4, 4):
            raise ValueError(
                f"prepare_frames expects frames of shape (..., N, 4, 4), "
                f"got {tuple(frames.shape)}"
            )
        self.frames = frames
        if not frames.is_global:
            inv_gamma = None
            if self.preserve_variance:
                if p_ref is None:
                    raise ValueError("preserve_variance requires `p_ref` in prepare_frames.")
                gamma = self._compute_gamma(frames, p_ref, ptr=ptr)
                # (..., 1, N, 1, 1): broadcasts over heads and the 4x4 matrix. Folded directly
                # into the frame matrices (see _scale_frames) rather than applied per-channel
                # to every q/k/v/output tensor in every layer, since a grade-n tensor transform
                # applies the frame matrix n times: scaling the matrix by 1/gamma is equivalent
                # to, but far cheaper than, dividing the post-transform tensor by gamma**grade.
                inv_gamma = (1 / gamma)[..., None, :, None, None]

            # insert frames head dimension
            frames_out = frames.reshape(*frames.shape[:-3], 1, frames.shape[-3], 4, 4)
            frames_out = frames_out.expand(
                *frames.shape[:-3], self.num_heads, frames.shape[-3], 4, 4
            )

            # create inv_frames and lower_inv_frames
            inv_frames = InverseFrames(frames_out)
            lower_inv_frames = LowerIndicesFrames(inv_frames)

            if self.preserve_variance:
                # rescale the pre-attention (local->global) q/k/v transform
                inv_frames = _scale_frames(inv_frames, inv_gamma)
                lower_inv_frames = _scale_frames(lower_inv_frames, inv_gamma)

            # qkv = (inv_frames, lower_inv_frames, inv_frames)
            # note that (lower_inv_frames, inv_frames, inv_frames) is equivalent
            self.frames_qkv = Frames(
                matrices=torch.cat(
                    [
                        inv_frames.matrices,
                        lower_inv_frames.matrices,
                        inv_frames.matrices,
                    ],
                    dim=0,
                ),
                is_identity=inv_frames.is_identity,
                is_global=inv_frames.is_global,
                det=torch.cat([inv_frames.det, lower_inv_frames.det, inv_frames.det], dim=0),
                parity=torch.cat(
                    [inv_frames.parity, lower_inv_frames.parity, inv_frames.parity], dim=0
                ),
                inv=torch.cat([inv_frames.inv, lower_inv_frames.inv, inv_frames.inv], dim=0),
            )

            if self.preserve_variance:
                # rescale the post-attention (global->local) output transform
                frames_out = _scale_frames(frames_out, inv_gamma)

            # flatten frames (preparation for tensorreps_transform)
            self.frames_out = frames_out.reshape(-1, 4, 4)
            self.frames_qkv = self.frames_qkv.reshape(-1, 4, 4)

    def _local_to_global(self, q_local, k_local, v_local):
        # check input shapes
        assert k_local.shape == v_local.shape == q_local.shape  # has to match perfectly
        assert 3 * prod(k_local.shape[:-1]) == self.frames_qkv.shape[-3]

        # transform q, k, v into global frame (preserve_variance rescaling, if enabled, is
        # already folded into self.frames_qkv, see prepare_frames)
        qkv_local = torch.cat([q_local, k_local, v_local], dim=0)
        qkv_global = self.transform(qkv_local, self.frames_qkv)
        q_global, k_global, v_global = qkv_global.chunk(3, dim=0)
        return q_global, k_global, v_global

    def _global_to_local(self, out_global):
        # transform result back into local frame (preserve_variance rescaling, if enabled,
        # is already folded into self.frames_out, see prepare_frames)
        return self.transform(out_global, self.frames_out)

    @staticmethod
    def _attention(query, key, value, **attn_kwargs):
        # (B, H, N, C) format required for scaled_dot_product_attention, so flatten any
        # extra leading dimensions and restore them afterwards
        shape_q, shape_k = query.shape, key.shape
        query = query.reshape(-1, *shape_q[-3:])
        key = key.reshape(-1, *shape_k[-3:])
        value = value.reshape(-1, *shape_k[-3:])

        out = scaled_dot_product_attention(query, key, value, **attn_kwargs)
        return out.view(*shape_q)  # (..., H, N, C)

    def forward(self, q_local, k_local, v_local, **attn_kwargs):
        """Execute LLoCa attention.

        Strategy
        1) Transform q, k, v into global frame
        2) Apply attention in global frame
        3) Transform output back into local frame

        Comments
        - Dimensions: ... (optional), H (head), N (particles), C (channels).
        - Extension to cross-attention is trivial but we don't have this right now for convenience. Strategy: frames_q for queries (in contrast to frames=frames_kv).

        Parameters
        ----------
        q_local: torch.tensor
            Local queries of shape (..., H, N, C)
        k_local: torch.tensor
            Local keys of shape (..., H, N, C)
        v_local: torch.tensor
            Local values of shape (..., H, N, C)
        **attn_kwargs
            Optional arguments that are passed on to the attention backend

        Returns
        -------
        out_local: torch.tensor
            Attention output in local frame of shape (..., H, N, C)
        """
        if self.frames.is_global:
            # fallback to standard attention for global frames
            return self._attention(q_local, k_local, v_local, **attn_kwargs)

        q_global, k_global, v_global = self._local_to_global(q_local, k_local, v_local)

        # attention (in global frame)
        out_global = self._attention(q_global, k_global, v_global, **attn_kwargs)

        out_local = self._global_to_local(out_global)
        return out_local


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    **attn_kwargs,
) -> Tensor:
    """Execute scaled dot-product attention.
    The attention backend is determined dynamically
    based on the ``**attn_kwargs``.

    Parameters
    ----------
    query : torch.Tensor
        Tensor of shape (..., items_out, channels)
    key : torch.Tensor
        Tensor of shape (..., items_in, channels)
    value : torch.Tensor
        Tensor of shape (..., items_in, channels)
    **attn_kwargs
        Optional keyword arguments passed to attention.

    Returns
    -------
    torch.Tensor
        Tensor of shape (..., head, item_out, channels)
    """
    backend = attn_kwargs.pop("backend", None)
    attention_backend = get_attention_backend(backend=backend, **attn_kwargs)
    return attention_backend(query, key, value, **attn_kwargs)
