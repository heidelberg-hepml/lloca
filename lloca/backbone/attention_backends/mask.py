"""Block-diagonal attention masks in the layout each backend expects."""

import torch

from . import SPARSE_BACKENDS, _backend_unavailable_message, _resolve_backend


def get_sparse_attention_mask(
    batch: torch.Tensor,
    attention_backend: str,
    dtype: torch.dtype,
):
    """Returns sparse attention mask according to the backend.

    Parameters
    ----------
    batch : torch.Tensor
        Batch vector, maps each token to its sequence in the batch.
    attention_backend : str
        Attention backend to use ("varlen", "xformers", "flex", or "flash").
    dtype : torch.dtype
        Data type of the attention mask (for xformers backend).

    Returns
    -------
    dict[str, torch.Tensor | BlockMask | BlockDiagonalMask]
        Attention mask for the specified backend.
    """
    assert attention_backend in SPARSE_BACKENDS, (
        f"attention_backend={attention_backend} does not support sparse representations, should be one of {SPARSE_BACKENDS}"
    )

    on_cpu = batch.device == torch.device("cpu")
    if on_cpu and attention_backend in {"xformers", "flash", "varlen"}:
        # These have no CPU kernel (xformers and flash are not even registered on CPU),
        # so fall back to dense attention with an additive block-diagonal mask. Built
        # directly rather than via xformers, which is unavailable here by construction.
        # flex is excluded: its block mask works on CPU.
        blockdiag = batch.unsqueeze(-1) == batch.unsqueeze(-2)
        mask = torch.zeros_like(blockdiag, dtype=dtype).masked_fill_(~blockdiag, float("-inf"))
        return {"attn_mask": mask}

    module = _resolve_backend(attention_backend)
    if module is None:
        raise ValueError(
            f"{_backend_unavailable_message(attention_backend)} "
            f"Run 'pip install lloca[{attention_backend}-attention]'."
        )
    if attention_backend == "xformers":
        bincounts = torch.bincount(batch).tolist()
        return {"attn_bias": module.BlockDiagonalMask.from_seqlens(bincounts)}
    elif attention_backend in {"flash", "varlen"}:
        seqlens = torch.bincount(batch).to(torch.int32)
        maxlen = int(seqlens.max().item())
        cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=seqlens.device), cu_seqlens], dim=0
        )
        if attention_backend == "flash":
            return {
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": maxlen,
                "max_seqlen_k": maxlen,
            }
        return {
            "cu_seq_q": cu_seqlens,
            "cu_seq_k": cu_seqlens,
            "max_q": maxlen,
            "max_k": maxlen,
        }
    else:  # flex
        N = batch.size(0)

        def jagged_masking(b, h, q_idx, kv_idx):
            return batch[q_idx] == batch[kv_idx]

        mask = module.create_block_mask(
            jagged_masking, None, None, N, N, device=batch.device, _compile=True
        )
        return {"block_mask": mask}
