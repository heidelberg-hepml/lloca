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
            by 1/gamma_i^grade to prevent the variance blowup from large boosts. Needs the 4-momenta
            in :meth:`prepare_frames`.
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

    def _compute_gamma(self, frames, fourmomenta, mask=None, ptr=None):
        """Invariant per-particle Lorentz factor gamma_i >= 1 that prevents variance blowup."""
        dtype = torch.promote_types(fourmomenta.dtype, torch.float32)
        L = frames.matrices.to(dtype)
        p_global = torch.einsum("...nij,...nj->...ni", frames.inv.to(dtype), fourmomenta.to(dtype))
        if mask is not None:
            # where (not *): padded particles contribute 0 even if their momenta are non-finite
            p_global = torch.where(mask[..., None], p_global, torch.zeros_like(p_global))
        # reference momentum: total momentum of the event (dense) or of each jet (ptr)
        if ptr is None:
            p_ref = p_global.sum(dim=-2, keepdim=True).expand_as(p_global)
        else:
            flat = p_global.reshape(-1, 4)
            seg = get_batch_from_ptr(ptr, num_items=flat.shape[0])
            p_ref = flat.new_zeros(ptr.numel() - 1, 4).index_add_(0, seg, flat)
            p_ref = p_ref.index_select(0, seg).reshape_as(p_global)
        m_ref = torch.sqrt(self.variance_eps**2 + lorentz_squarednorm(p_ref).clamp(min=0))
        gamma = torch.einsum("...nij,...nj->...ni", L, p_ref)[..., 0] / m_ref
        return gamma.detach()  # fixed normalization: no gradient into the frames

    @minimum_autocast_precision(torch.float32)
    def prepare_frames(self, frames, fourmomenta=None, mask=None, ptr=None):
        """Prepare local frames for LLoCa attention (called once per forward pass).

        Parameters
        ----------
        frames: Frames
            Local frames of shape (..., N, 4, 4).
        fourmomenta: torch.tensor, optional
            Local per-particle 4-momenta (..., N, 4), energy-first; required when the
            ``preserve_variance`` flag is on, ignored otherwise.
        mask: torch.tensor, optional
            Real-particle mask (..., N); None means all real.
        ptr: torch.tensor, optional
            Jet boundaries for a packed layout; the reference momentum is then per jet.
        """
        self.frames = frames
        if not frames.is_global:
            inv_gamma = None
            if self.preserve_variance:
                if fourmomenta is None:
                    raise ValueError("preserve_variance requires `fourmomenta` in prepare_frames.")
                gamma = self._compute_gamma(
                    frames, fourmomenta, None if mask is None else mask.bool(), ptr=ptr
                )
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
            return scaled_dot_product_attention(
                q_local,
                k_local,
                v_local,
                **attn_kwargs,
            )

        q_global, k_global, v_global = self._local_to_global(q_local, k_local, v_local)

        # (B, H, N, C) format required for scaled_dot_product_attention
        shape_q, shape_k = q_global.shape, k_global.shape
        q_global = q_global.reshape(-1, *shape_q[-3:])
        k_global = k_global.reshape(-1, *shape_k[-3:])
        v_global = v_global.reshape(-1, *shape_k[-3:])

        # attention (in global frame)
        out_global = scaled_dot_product_attention(
            q_global,
            k_global,
            v_global,
            **attn_kwargs,
        )

        out_global = out_global.view(*shape_q)  # (..., H, N, C)

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
    attention_backend = get_attention_backend(**attn_kwargs)
    return attention_backend(query, key, value, **attn_kwargs)
