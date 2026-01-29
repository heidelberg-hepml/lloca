import torch

from . import _REGISTRY, SPARSE_BACKENDS


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
    assert attention_backend in _REGISTRY, (
        f"attention_backend={attention_backend} not installed, run 'pip install lloca[{attention_backend}]'"
    )

    def get_cpu_blockdiag_mask(batch):
        # materialize mask to torch.tensor (only for testing purposes)
        bincounts = torch.bincount(batch).tolist()
        blockdiag_fn = _REGISTRY["xformers"].BlockDiagonalMask
        mask = blockdiag_fn.from_seqlens(bincounts)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device, dtype=dtype)
        return mask

    on_cpu = batch.device == torch.device("cpu")
    if attention_backend == "xformers":
        if not on_cpu:
            bincounts = torch.bincount(batch).tolist()
            blockdiag_fn = _REGISTRY["xformers"].BlockDiagonalMask
            mask = blockdiag_fn.from_seqlens(bincounts)
            return {"attn_bias": mask}
        else:
            # fallback to default attention
            mask = get_cpu_blockdiag_mask(batch)
            return {"attn_mask": mask}
    elif attention_backend in {"flash", "varlen"}:
        seqlens = torch.bincount(batch).to(torch.int32)
        maxlen = int(seqlens.max().item())
        cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=seqlens.device), cu_seqlens], dim=0
        )
        if not on_cpu:
            if attention_backend == "flash":
                return {
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": maxlen,
                    "max_seqlen_k": maxlen,
                }
            elif attention_backend == "varlen":
                return {
                    "cu_seq_q": cu_seqlens,
                    "cu_seq_k": cu_seqlens,
                    "max_q": maxlen,
                    "max_k": maxlen,
                }
        else:
            # fallback to default attention
            mask = get_cpu_blockdiag_mask(batch)
            return {"attn_mask": mask}
    elif attention_backend == "flex":
        N = batch.size(0)

        def jagged_masking(b, h, q_idx, kv_idx):
            return batch[q_idx] == batch[kv_idx]

        block_fn = _REGISTRY["flex"].create_block_mask
        mask = block_fn(jagged_masking, None, None, N, N, device=batch.device, _compile=True)
        return {"block_mask": mask}
    else:
        raise ValueError(
            f"Unsupported attention backend: {attention_backend}. "
            'Supported backends are "varlen", "xformers", "flex", and "flash".'
        )
