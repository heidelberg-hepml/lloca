"""Tests for the attention backend dispatch and the sparse attention masks."""

import pytest
import torch

from lloca.backbone.attention import scaled_dot_product_attention
from lloca.backbone.attention_backends import SPARSE_BACKENDS, get_attention_backend
from lloca.backbone.attention_backends.mask import get_sparse_attention_mask
from tests.constants import STRICT_TOLERANCES


def test_dispatch_defaults_to_native():
    """Without backend-specific kwargs we get PyTorch's own attention."""
    assert get_attention_backend() is torch.nn.functional.scaled_dot_product_attention


def test_explicit_backend_is_not_forwarded_to_the_kernel():
    """``backend=...`` selects the kernel; it must not be passed on to the kernel itself.

    It used to be forwarded along with the other attn_kwargs, so every explicit backend
    selection raised a TypeError inside the backend.
    """
    q, k, v = (torch.randn(1, 2, 5, 8, dtype=torch.float64) for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, backend="native"), expected, **STRICT_TOLERANCES
    )


def test_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown attention backend"):
        get_attention_backend(backend="not-a-backend")


# flex is the one sparse backend with a CPU kernel: it returns a block mask rather than
# the dense fallback, so it has nothing to compare here.
DENSE_FALLBACK_BACKENDS = sorted(set(SPARSE_BACKENDS) - {"flex"})


@pytest.mark.parametrize("attention_backend", DENSE_FALLBACK_BACKENDS)
def test_sparse_mask_matches_dense_attention_on_cpu(attention_backend):
    """Every dense-fallback mask must reproduce dense block-diagonal attention.

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
    got = torch.nn.functional.scaled_dot_product_attention(q, k, v, **attn_kwargs)
    torch.testing.assert_close(got, expected, **STRICT_TOLERANCES)


def test_sparse_mask_rejects_unknown_backend():
    batch = torch.tensor([0, 0, 1])
    with pytest.raises(AssertionError):
        get_sparse_attention_mask(batch, "not-a-backend", torch.float32)
