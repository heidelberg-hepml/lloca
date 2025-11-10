import torch

from lloca.equivectors.mlp import MLPVectors
from lloca.utils.lorentz import lorentz_metric


def sample_particle(shape, logm2_std, logm2_mean, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cpu")
    assert logm2_std > 0
    logm2 = torch.randn(*shape, 1, device=device, dtype=dtype) * logm2_std + logm2_mean
    p3 = torch.randn(*shape, 3, device=device, dtype=dtype)
    E = torch.sqrt(logm2.exp() + (p3**2).sum(dim=-1, keepdim=True))
    return torch.cat([E, p3], dim=-1)


def lorentz_test(trafo, **kwargs):
    """
    Test that the transformation matrix T is orthogonal

    Condition: T^T * g * T = g
    with the Lorentz metric g = diag(1, -1, -1, -1)
    """
    metric = lorentz_metric(trafo.shape[:-2], trafo.device, trafo.dtype)
    test = torch.einsum("...ij,...jk,...kl->...il", trafo, metric, trafo.transpose(-1, -2))
    torch.testing.assert_close(test, metric, **kwargs)


def equivectors_builder(num_scalars=0):
    def builder(n_vectors):
        return MLPVectors(
            n_vectors=n_vectors,
            num_scalars=num_scalars,
            hidden_channels=16,
            num_layers_mlp=1,
        )

    return builder
