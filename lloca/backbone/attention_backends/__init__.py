"""Dynamic attention backend selection."""

from functools import lru_cache
from importlib import metadata

import torch

# common kwargs used in custom attention backends
XFORMERS_KWARGS = ["attn_bias", "op"]
FLEX_KWARGS = ["score_mod", "block_mask"]
FLASH_KWARGS = ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"]


@lru_cache
def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


_REGISTRY = {}
for ep in metadata.entry_points(group="lloca.backbone.attention_backends"):
    try:
        # check if entry point code be loaded without ImportError
        module = ep.load()
    except ImportError:
        continue

    if ep.name in ["xformers", "flash"] and get_device() == torch.device("cpu"):
        # xformers and flash-attn are not available on CPU
        continue
    _REGISTRY[ep.name] = module


def get_sparse_attention_mask(*args, **kwargs):
    """
    Wrapper function to access xformers attention mask function if available.
    """
    try:
        return _REGISTRY["xformers_attention"].get_xformers_attention_mask(*args, **kwargs)
    except KeyError as err:
        raise RuntimeError(
            "Sparse attention masks currently only supported using xformers. "
            "Run 'pip install lloca[xformers_attention]'."
        ) from err


def get_attention_backend(**kwargs):
    """
    Dynamically determine the attention backend based on the extra keyword arguments.

    Implemented backends:
    - PyTorch's native attention: torch.nn.functional.scaled_dot_product_attention
    - Xformers attention: xformers.ops.memory_efficient_attention
    - PyTorch's flex_attention: torch.nn.attention.flex_attention.flex_attention
    - Original flash attention (supports variable sequence length): flash_attn.flash_attn_varlen_func
    """
    # check if backend is explicitly specified
    backend = kwargs.get("backend", None)
    if backend in _REGISTRY:
        return _REGISTRY[backend].attention

    # automatic fall-back based on other **kwargs
    if any(kwargs.get(kwarg, None) is not None for kwarg in XFORMERS_KWARGS):
        return _REGISTRY["xformers"].attention
    elif any(kwargs.get(kwarg, None) is not None for kwarg in FLEX_KWARGS):
        return _REGISTRY["flex"].attention
    elif any(kwargs.get(kwarg, None) is not None for kwarg in FLASH_KWARGS):
        return _REGISTRY["flash"].attention

    # fall-back to native torch attention
    try:
        return _REGISTRY["native"].attention
    except KeyError as err:
        raise RuntimeError(
            f"No attention backend could be resolved. Available backends: {list(_REGISTRY)}"
        ) from err
