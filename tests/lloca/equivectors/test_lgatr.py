import importlib.util

import pytest
import torch
from lgatr.nets import LGATr

from lloca.backbone.attention_backends import _REGISTRY
from lloca.equivectors.lgatr import LGATrVectors
from lloca.utils.rand_transforms import rand_lorentz
from tests.constants import LOGM2_MEAN_STD, TOLERANCES
from tests.helpers import sample_particle

_cuda_available = torch.cuda.is_available()
_xformers_available = importlib.util.find_spec("xformers") is not None
_flash_available = importlib.util.find_spec("flash-attn") is not None
_torch_version = torch.__version__.split("+")[0]
_flex_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 7)
_varlen_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 10)


@pytest.mark.parametrize("batch_dims", [[100]])
@pytest.mark.parametrize("jet_size", [10])
@pytest.mark.parametrize("n_vectors", [1, 2, 3])
@pytest.mark.parametrize("num_blocks,hidden_mv_channels,hidden_s_channels", [(1, 2, 8)])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("lgatr_norm", [True, False])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
@pytest.mark.parametrize("num_scalars", [0, 1])
@pytest.mark.parametrize(
    "sparse_mode, attention_backend",
    [(False, None), (True, "xformers"), (True, "flex"), (True, "flash"), (True, "varlen")],
)
def test_equivariance(
    batch_dims,
    jet_size,
    n_vectors,
    hidden_mv_channels,
    hidden_s_channels,
    num_blocks,
    layer_norm,
    lgatr_norm,
    logm2_std,
    logm2_mean,
    num_scalars,
    sparse_mode,
    attention_backend,
):
    if (
        (attention_backend == "xformers" and not (_xformers_available and _cuda_available))
        or (attention_backend == "flex" and not _flex_available)
        or (attention_backend == "flash" and not (_flash_available and _cuda_available))
        or (attention_backend == "varlen" and not (_varlen_available and _cuda_available))
        or attention_backend not in _REGISTRY
    ):
        return

    assert len(batch_dims) == 1
    dtype = torch.float64

    def builder(in_s_channels, out_mv_channels, out_s_channels):
        return LGATr(
            in_mv_channels=1,
            attention={},
            mlp={},
            in_s_channels=in_s_channels,
            out_mv_channels=out_mv_channels,
            out_s_channels=out_s_channels,
            hidden_mv_channels=hidden_mv_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
        )

    # construct sparse tensors containing a set of equal-multiplicity jets
    ptr = torch.arange(0, (batch_dims[0] + 1) * jet_size, jet_size) if sparse_mode else None

    # input to mlp: only edge attributes
    def calc_node_attr(fm):
        return torch.zeros(*fm.shape[:-1], num_scalars, dtype=dtype)

    equivectors = LGATrVectors(
        net=builder,
        n_vectors=n_vectors,
        num_scalars=num_scalars,
        hidden_mv_channels=hidden_mv_channels,
        hidden_s_channels=hidden_s_channels,
        layer_norm=layer_norm,
        lgatr_norm=lgatr_norm,
        attention_backend=attention_backend,
    ).to(dtype=dtype)

    fm_test = sample_particle(batch_dims + [jet_size], logm2_std, logm2_mean, dtype=dtype)
    if sparse_mode:
        fm_test = fm_test.flatten(0, 1)
    equivectors.init_standardization(fm_test, ptr=ptr)

    fm = sample_particle(batch_dims + [jet_size], logm2_std, logm2_mean, dtype=dtype)
    if sparse_mode:
        fm = fm.flatten(0, 1)

    # careful: same global transformation for each jet
    random = rand_lorentz(batch_dims, dtype=dtype)
    random = random.unsqueeze(1).repeat(1, jet_size, 1, 1).view(*fm.shape, 4)

    # path 1: global transform + predict vectors
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    node_attr_prime = calc_node_attr(fm_prime)
    vecs_prime1 = equivectors(fourmomenta=fm_prime, scalars=node_attr_prime, ptr=ptr)

    # path 2: predict vectors + global transform
    node_attr = calc_node_attr(fm)
    vecs = equivectors(fourmomenta=fm, scalars=node_attr, ptr=ptr)
    vecs_prime2 = torch.einsum("...ij,...kj->...ki", random, vecs)

    # test that vectors are predicted equivariantly
    torch.testing.assert_close(vecs_prime1, vecs_prime2, **TOLERANCES)
