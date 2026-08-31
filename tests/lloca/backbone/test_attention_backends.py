"""Tests for the attention backend dispatch and the sparse attention masks."""

import pytest
import torch

from lloca.backbone.attention_backends import SPARSE_BACKENDS, get_attention_backend
from lloca.backbone.attention_backends.mask import get_sparse_attention_mask
from tests.constants import STRICT_TOLERANCES


def test_dispatch_defaults_to_native():
    """Without backend-specific kwargs we get PyTorch's own attention."""
    assert get_attention_backend() is torch.nn.functional.scaled_dot_product_attention


@pytest.mark.parametrize("attention_backend", SPARSE_BACKENDS)
def test_sparse_mask_matches_dense_attention_on_cpu(attention_backend):
    """Every sparse mask must reproduce dense block-diagonal attention.

    On CPU the backends without a CPU kernel fall back to an additive dense mask; this test
    pins that the fallback is equivalent to masking by hand, which is the property the
    fallback exists to provide.
    """
    torch.manual_seed(0)
    batch = torch.tensor([0, 0, 0, 1, 1, 2])
    n = batch.numel()
    q, k, v = (torch.randn(1, 2, n, 8, dtype=torch.float64) for _ in range(3))

    blockdiag = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=blockdiag)

    attn_kwargs = get_sparse_attention_mask(batch, attention_backend, torch.float64)
    if "attn_mask" not in attn_kwargs:
        pytest.skip(f"{attention_backend} does not use the dense CPU fallback here")

    got = torch.nn.functional.scaled_dot_product_attention(q, k, v, **attn_kwargs)
    torch.testing.assert_close(got, expected, **STRICT_TOLERANCES)


def test_sparse_mask_rejects_unknown_backend():
    batch = torch.tensor([0, 0, 1])
    with pytest.raises(AssertionError):
        get_sparse_attention_mask(batch, "not-a-backend", torch.float32)
