"""Tests for the self-contained μP support in :mod:`lloca.mup`.

The decisive test is :func:`test_coord_check_mlp`: it trains the ``MLP`` backbone at
a range of widths for a few steps on a *fixed* batch and checks that the activation
coordinates stay O(1) in width under μP (near-flat) but drift under standard
parametrization (SP). This is the standard μP "coordinate check". The MLP backbone is
used because it needs no frames, so the check is fast and self-contained on CPU.
"""

import numpy as np
import pytest
import torch
from torch import nn

mup = pytest.importorskip("mup")

from lloca.backbone.mlp import MLP  # noqa: E402
from lloca.backbone.transformer import Transformer  # noqa: E402
import lloca.mup as lmup  # noqa: E402


def _make_mlp(hidden, parametrization):
    return MLP(
        in_shape=[8],
        out_shape=[4],
        hidden_channels=hidden,
        hidden_layers=3,
        parametrization=parametrization,
    )


def test_sp_path_unchanged():
    """parametrization='sp' (default) leaves the plain modules and no infshapes."""
    m = _make_mlp(64, "sp")
    assert isinstance(m.mlp[-1], nn.Linear) and not isinstance(m.mlp[-1], mup.MuReadout)
    assert not any(hasattr(p, "infshape") for p in m.parameters())

    t = Transformer(
        in_channels=10, attn_reps="4x0n+1x1n", out_channels=4, num_blocks=2, num_heads=4
    )
    assert type(t.linear_out).__name__ == "Linear"
    assert not any(hasattr(p, "infshape") for p in t.parameters())


def test_mup_base_shapes_set_automatically():
    """parametrization='mup' sets infshapes whose width multipliers scale with width."""
    m = _make_mlp(256, "mup")
    assert isinstance(m.mlp[-1], mup.MuReadout)
    assert all(hasattr(p, "infshape") for p in m.parameters())

    # base width default is 64; the first hidden layer's out dim (=hidden) is the
    # width axis, so its width multiplier is hidden / base.
    first = m.mlp[0]
    assert pytest.approx(first.weight.infshape.width_mult()) == 256 / 64


@pytest.mark.parametrize("backbone_width", [4, 8, 16])
def test_transformer_mup_width_multiplier(backbone_width):
    """Transformer width axis is num_heads; base default is 2."""
    t = Transformer(
        in_channels=10,
        attn_reps="4x0n+1x1n",
        out_channels=4,
        num_blocks=2,
        num_heads=backbone_width,
        parametrization="mup",
    )
    assert isinstance(t.linear_out, mup.MuReadout)
    assert pytest.approx(t.linear_in.weight.infshape.width_mult()) == backbone_width / 2


def test_finalize_marks_external_params_sp():
    """finalize() stamps an SP (width_mult 1) infshape on params outside μP backbones."""

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(5, 8)  # external, fixed width -> SP
            self.net = _make_mlp(128, "mup")

    w = Wrapper()
    assert not hasattr(w.encoder.weight, "infshape")
    lmup.finalize(w)
    assert w.encoder.weight.infshape.width_mult() == 1
    assert all(hasattr(p, "infshape") for p in w.parameters())
    # a single μP optimizer can now handle the whole (mixed) model
    lmup.MuAdamW(w.parameters(), lr=1e-3)


def test_reload_reproduces_base_shapes():
    """Reconstructing with the same args reproduces identical base shapes, so a
    checkpoint round-trip needs no extra μP bookkeeping."""
    m1 = _make_mlp(256, "mup")
    m2 = _make_mlp(256, "mup")
    sd = m1.state_dict()
    m2.load_state_dict(sd)  # load_state_dict must not disturb infshapes
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert p1.infshape == p2.infshape, n1
        assert torch.equal(p1.detach(), p2.detach())


def _coord_check(mode, widths, steps=4, seed=0):
    """Return the mean per-coordinate magnitude of the pre-readout activations at
    the final step, for each width."""
    torch.manual_seed(seed)
    x = torch.randn(16, 8)
    y = torch.randn(16, 4)

    magnitudes = []
    for hidden in widths:
        torch.manual_seed(seed)  # same data/init seed across widths
        model = _make_mlp(hidden, mode)
        if mode == "mup":
            lmup.finalize(model)
            opt = lmup.MuAdam(model.parameters(), lr=1e-2)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        captured = {}

        def hook(_module, inp, _out):
            captured["act"] = inp[0].detach()

        handle = model.mlp[-1].register_forward_hook(hook)
        for _ in range(steps):
            opt.zero_grad()
            out = model(x)
            loss = ((out - y) ** 2).mean()
            loss.backward()
            opt.step()
        handle.remove()
        magnitudes.append(captured["act"].abs().mean().item())
    return np.array(magnitudes)


def test_coord_check_mlp():
    """μP keeps activation coordinates ~flat in width; SP lets them drift.

    We compare the slope of log(mean|activation|) vs log(width). μP should be close
    to flat; SP should have a clearly larger-magnitude slope.
    """
    widths = [64, 128, 256, 512, 1024]
    log_w = np.log(np.array(widths, dtype=float))

    mup_mag = _coord_check("mup", widths)
    sp_mag = _coord_check("sp", widths)

    mup_slope = np.polyfit(log_w, np.log(mup_mag), 1)[0]
    sp_slope = np.polyfit(log_w, np.log(sp_mag), 1)[0]

    # μP coordinates are approximately width-invariant ...
    assert abs(mup_slope) < 0.25, f"μP slope {mup_slope:.3f} not flat (mags={mup_mag})"
    # ... and markedly flatter than SP, which drifts with width.
    assert abs(mup_slope) < abs(sp_slope), (
        f"μP slope {mup_slope:.3f} not flatter than SP {sp_slope:.3f}"
    )
