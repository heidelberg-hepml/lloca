"""Self-contained Maximal-Update Parametrization (μP) support for LLoCa backbones.

μP (Yang & Hu, 2022) makes the optimal hyperparameters of a network (most
importantly the learning rate) *transfer* across widths, so a configuration
tuned on a small model can be reused as the model is scaled up. The standard
``mup`` workflow is fiddly: the user must build two extra "base" and "delta"
copies of the model, call :func:`mup.make_base_shapes` / :func:`mup.set_base_shapes`,
manage a ``.bsh`` file, and re-apply the base shapes after every checkpoint load.

This module hides all of that behind a single ``parametrization="mup"`` flag on
the backbones. The key observation is that, for a LLoCa backbone, the base shapes
are a *deterministic function of the constructor arguments*: the "width" axis
(e.g. ``num_heads`` for the transformer, ``hidden_channels`` for the MLP) is just
another argument. So a backbone can build its own base/delta copies in memory and
compute its base shapes inside ``__init__`` -- no files, no user bookkeeping, and
reloading a checkpoint (same arguments -> identical base shapes) "just works".

Public surface (see :mod:`lloca.mup`):

* decorate a backbone with :func:`mup_parametrized` and give it a
  ``parametrization`` keyword-only argument (plus optional ``mup_base_shapes`` /
  ``mup_delta_shapes`` overrides),
* build its readout with :func:`make_readout`,
* call :func:`finalize` once on the *full* training model (so that any parameter
  living outside a μP backbone -- e.g. a frames network -- is marked as standard
  parametrization), and optimize with :class:`lloca.mup.MuAdam` /
  :class:`~lloca.mup.MuAdamW`.

Only :class:`~lloca.backbone.transformer.Transformer` and
:class:`~lloca.backbone.mlp.MLP` are validated by the coordinate-check test
(``tests/lloca/mup``); the other backbones carry the same mechanical wiring but
their μP-correctness has not been verified -- treat with care.
"""

import functools
import inspect
import threading

import torch
from torch import nn

try:
    from mup import MuReadout, make_base_shapes, set_base_shapes
    from mup.infshape import InfDim, InfShape

    _MUP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when mup is missing
    MuReadout = None
    _MUP_AVAILABLE = False


# Constructor keywords used to drive μP; excluded when cloning a backbone into its
# base/delta variants (those are forced explicitly instead).
_MUP_CONTROL_KWARGS = ("parametrization", "mup_base_shapes", "mup_delta_shapes")

# Re-entrancy guard: while a backbone builds its base/delta copies, the decorated
# __init__ of those copies must *not* recurse into base-shape computation.
_build_state = threading.local()


def mup_available() -> bool:
    """Return True if the optional ``mup`` dependency is importable."""
    return _MUP_AVAILABLE


def _require_mup():
    if not _MUP_AVAILABLE:
        raise ImportError(
            "parametrization='mup' requires the optional 'mup' package. "
            "Install it with `pip install lloca[mup]` (or `pip install mup`)."
        )


class _building_base_shapes:
    """Context manager marking that we are inside base/delta construction."""

    def __enter__(self):
        self._prev = getattr(_build_state, "active", False)
        _build_state.active = True

    def __exit__(self, *exc):
        _build_state.active = self._prev


def _is_building_base_shapes() -> bool:
    return getattr(_build_state, "active", False)


def is_mup(module: nn.Module) -> bool:
    """Whether ``module`` was constructed with ``parametrization='mup'``."""
    return getattr(module, "parametrization", "sp") == "mup"


def make_readout(in_features: int, out_features: int, parametrization: str = "sp", **kwargs):
    """Return the output projection appropriate for ``parametrization``.

    For ``"mup"`` this is a :class:`mup.MuReadout` (which applies the μP ``1/width``
    output scaling and supports zero-initialization); for ``"sp"`` it is a plain
    :class:`torch.nn.Linear`, so the standard-parametrization code path is byte-for-byte
    unchanged.
    """
    if parametrization == "mup":
        _require_mup()
        kwargs.setdefault("readout_zero_init", True)
        return MuReadout(in_features, out_features, **kwargs)
    return nn.Linear(in_features, out_features, **kwargs)


def normal_fanin_(linear: nn.Linear, gain: float = 1.0) -> None:
    """μP base init for a hidden/input linear: ``N(0, gain^2 / fan_in)`` weights.

    ``mup.set_base_shapes(..., rescale_params=True)`` later rescales these by the
    appropriate width multiplier, so the *base* init must be a plain fan-in normal.
    """
    fan_in = linear.weight.shape[1]
    nn.init.normal_(linear.weight, mean=0.0, std=gain * fan_in**-0.5)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def reinitialize_mup(module: nn.Module, gain: float = 1.0) -> None:
    """Apply a generic μP base initialization to every linear in ``module``.

    Hidden/input :class:`~torch.nn.Linear` layers get a fan-in normal; readouts
    (:class:`mup.MuReadout`) are zero-initialized. This is the default recipe used
    by backbones that do not override their own ``_mup_reset_parameters``.
    """
    for sub in module.modules():
        if _MUP_AVAILABLE and isinstance(sub, MuReadout):
            nn.init.zeros_(sub.weight)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.Linear):
            normal_fanin_(sub, gain=gain)


def _resolve_override(override_value, current_value):
    """An override may be a static value or a callable ``current -> new``.

    The callable form lets width axes that are not plain integers be scaled
    relative to the target model (e.g. ``{"hidden_reps": lambda r: scale_reps(r, 0.5)}``).
    """
    return override_value(current_value) if callable(override_value) else override_value


def _rebuild_variant(cls, sig, bound, overrides):
    """Re-instantiate ``cls`` from a previous call's bound arguments + ``overrides``.

    Reconstructs the original call faithfully (handling ``*args``/``**kwargs`` and
    keyword-only parameters), forces ``parametrization='mup'`` and drops the μP
    control kwargs, then applies the width ``overrides`` (e.g. ``{"num_heads": 2}``;
    values may be callables, see :func:`_resolve_override`). Runs under the
    re-entrancy guard so the clone does not recurse.
    """
    pos, var_pos, kw = [], (), {}
    for name, param in sig.parameters.items():
        if name == "self" or name in _MUP_CONTROL_KWARGS:
            continue
        if param.kind == param.VAR_POSITIONAL:
            var_pos = bound.arguments.get(name, ())
        elif param.kind == param.VAR_KEYWORD:
            kw.update(bound.arguments.get(name, {}))
        elif param.kind == param.POSITIONAL_OR_KEYWORD:
            current = bound.arguments.get(name, param.default)
            pos.append(_resolve_override(overrides[name], current) if name in overrides else current)
        elif param.kind == param.KEYWORD_ONLY:
            if name in overrides:
                kw[name] = _resolve_override(overrides[name], bound.arguments.get(name))
            elif name in bound.arguments:
                kw[name] = bound.arguments[name]
    # overrides that target **kwargs rather than a named parameter
    for key, value in overrides.items():
        if key not in sig.parameters:
            kw[key] = _resolve_override(value, kw.get(key))
    kw["parametrization"] = "mup"
    with _building_base_shapes():
        return cls(*pos, *var_pos, **kw)


def _apply_base_shapes(model):
    """Compute and set μP base shapes on ``model`` from its stored constructor call.

    Builds base- and delta-width copies in memory (deterministic from the
    constructor arguments), derives the base shapes with
    :func:`mup.make_base_shapes`, and applies them to ``model`` with
    ``rescale_params=True`` so the live parameters are μP-scaled in place.
    """
    _require_mup()
    cls = model._mup_class
    sig = model._mup_signature
    bound = model._mup_bound
    base_over, delta_over = model._mup_shape_overrides

    base_model = _rebuild_variant(cls, sig, bound, base_over)
    delta_model = _rebuild_variant(cls, sig, bound, delta_over)

    base_shapes = make_base_shapes(base_model, delta_model)
    set_base_shapes(model, base_shapes, rescale_params=True)
    model._mup_base_shapes = base_shapes


def mup_parametrized(cls):
    """Class decorator giving a backbone self-contained μP support.

    The decorated class must:

    * accept the keyword-only arguments ``parametrization`` (``"sp"`` | ``"mup"``),
      ``mup_base_shapes`` and ``mup_delta_shapes`` (each an optional ``dict`` of
      constructor-argument overrides defining the base/delta widths),
    * set ``self.parametrization`` during ``__init__``,
    * define a class attribute ``DEFAULT_MUP_SHAPES = (base_overrides, delta_overrides)``
      giving the default width overrides when the user does not supply them,
    * build its readout(s) with :func:`make_readout` and its μP init either via
      :func:`reinitialize_mup` or a custom ``_mup_reset_parameters`` method.

    When ``parametrization == "mup"`` and we are *not* already inside a base/delta
    build, the wrapped ``__init__`` records the construction arguments and computes
    the base shapes by re-instantiating the model at the base/delta widths.
    """
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_init(self, *args, **kwargs)

        if not is_mup(self) or _is_building_base_shapes():
            return

        base_over = bound.arguments.get("mup_base_shapes") or cls.DEFAULT_MUP_SHAPES[0]
        delta_over = bound.arguments.get("mup_delta_shapes") or cls.DEFAULT_MUP_SHAPES[1]
        self._mup_class = cls
        self._mup_signature = signature
        self._mup_bound = bound
        self._mup_shape_overrides = (dict(base_over), dict(delta_over))
        _apply_base_shapes(self)

    cls.__init__ = __init__
    return cls


def scale_reps(reps, factor: float, min_mul: int = 1) -> str:
    """Scale the multiplicities of a :class:`~lloca.reps.tensorreps.TensorReps` by ``factor``.

    Useful as a callable width-override for backbones whose width axis is a tensor
    representation (e.g. ``GraphNet``'s ``hidden_reps``): the rep *structure* (orders
    and parities) is preserved while every multiplicity is scaled, so the base/delta
    models differ only along the width axis. Returns a reps string.
    """
    from ..reps.tensorreps import TensorReps, _TensorMulRep

    reps = TensorReps(reps)
    scaled = [_TensorMulRep(max(min_mul, round(mr.mul * factor)), mr.rep) for mr in reps]
    return repr(TensorReps(scaled))


def finalize(model: nn.Module) -> nn.Module:
    """Mark every not-yet-covered parameter of ``model`` as standard parametrization.

    A μP backbone sets ``param.infshape`` on its own parameters during construction.
    Any *other* parameter in the full training model (e.g. a frames network or an
    input encoder that lives outside the backbone and whose width is fixed) is
    standard-parametrization; this stamps it with a trivial (finite) infshape so the
    μP optimizers treat it as a width multiplier of 1.

    Call this once after assembling the full model and before building the optimizer.
    Safe to call when ``mup`` is unavailable or no parameter needs it (no-op).
    """
    for p in model.parameters():
        if not hasattr(p, "infshape"):
            _require_mup()
            p.infshape = InfShape([InfDim(None, d) for d in p.shape])
    return model
