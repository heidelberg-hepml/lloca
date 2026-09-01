import importlib.util

import pytest
import torch

from lloca.backbone.attention_backends import _resolve_backend
from lloca.equivectors.mlp import MLPVectors
from lloca.utils.lorentz import lorentz_metric

_CUDA_AVAILABLE = torch.cuda.is_available()
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])

# Requirements per attention backend, on top of the backend module being importable.
_BACKEND_AVAILABLE = {
    "xformers": importlib.util.find_spec("xformers") is not None and _CUDA_AVAILABLE,
    "flex": _TORCH_VERSION >= (2, 7),
    "flash": importlib.util.find_spec("flash_attn") is not None and _CUDA_AVAILABLE,
    "varlen": _TORCH_VERSION >= (2, 10) and _CUDA_AVAILABLE,
}


def skip_if_backend_unavailable(attention_backend):
    """Skip the current test if ``attention_backend`` cannot run in this environment.

    ``attention_backend=None`` means dense attention, which is always available.
    Skipping (rather than returning) keeps unavailable backends visible in the pytest
    report instead of counting them as passed.
    """
    if attention_backend is None:
        return
    if not _BACKEND_AVAILABLE.get(attention_backend, False):
        pytest.skip(f"attention backend {attention_backend!r} is not available here")
    if _resolve_backend(attention_backend) is None:
        pytest.skip(f"attention backend {attention_backend!r} could not be loaded")


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


def _sweep_value_id(value):
    return getattr(value, "__name__", None) or str(value)


def _sweep_id(config, base):
    varied = {key: value for key, value in config.items() if base[key] != value}
    if not varied:
        return "base"
    return "-".join(f"{k}={_sweep_value_id(v)}" for k, v in varied.items())


def sweep(base, *axes):
    """Parametrization that varies one axis at a time instead of taking the full product.

    ``base`` is the default configuration as a dict, and every axis is an
    ``(argnames, argvalues)`` pair in the same spelling ``pytest.mark.parametrize`` uses;
    the result is the base configuration plus one configuration per axis value. The options
    these tests parametrize over are independent knobs, so a full cartesian product
    multiplies the runtime without covering any interaction the axes do not already cover;
    sweeping keeps every option value exercised at a cost that grows with the sum of the
    axis lengths rather than their product.

    Returns the ``(argnames, argvalues)`` pair to splat into ``pytest.mark.parametrize``.
    """
    configs = [base]
    for argnames, argvalues in axes:
        names = argnames.replace(" ", "").split(",")
        assert set(names) <= set(base), f"unknown axis {argnames!r}"
        for values in argvalues:
            override = dict(zip(names, values if len(names) > 1 else [values], strict=True))
            if all(base[key] == value for key, value in override.items()):
                continue  # already covered by the base configuration
            configs.append({**base, **override})

    names = list(base)
    params = [
        pytest.param(*[config[name] for name in names], id=_sweep_id(config, base))
        for config in configs
    ]
    return ",".join(names), params
