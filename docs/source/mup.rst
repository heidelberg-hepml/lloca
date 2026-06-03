Maximal Update Parametrization (μP)
===================================

`Maximal Update Parametrization <https://arxiv.org/abs/2203.03466>`_ (μP) rescales a
network's initialization, learning rate and output so that the *optimal*
hyperparameters become (approximately) independent of the model width. In practice
this means you can tune on a small model and reuse the configuration as you scale up,
instead of re-tuning at every size -- which is particularly valuable when a backbone is
sensitive to its hyperparameters as it grows.

``lloca`` ships a self-contained μP implementation behind a single
``parametrization`` flag. It is an optional feature; install the dependency with

.. code-block:: bash

   pip install lloca[mup]

Supported backbones: :class:`~lloca.backbone.transformer.Transformer`,
:class:`~lloca.backbone.transformer_v2.Transformer` (V2),
:class:`~lloca.backbone.mlp.MLP` and :class:`~lloca.backbone.graphnet.GraphNet`.

Quick start
-----------

.. code-block:: python

   import lloca.mup as mup
   from lloca.backbone import Transformer

   # 1. Build the backbone in μP mode. The width axis is num_heads; the base shapes
   #    are computed automatically inside __init__ -- no .bsh file, no base/delta models.
   net = Transformer(
       in_channels=in_channels,
       attn_reps="8x0n+2x1n",
       out_channels=out_channels,
       num_blocks=num_blocks,
       num_heads=num_heads,        # <- scale this freely; HPs transfer
       parametrization="mup",
   )

   # 2. If the trained model has parameters *outside* a μP backbone (e.g. a frames
   #    network or an input encoder, whose width is fixed), wrap everything and call
   #    finalize() once to mark those as standard parametrization.
   model = MyModel(net)            # net is a submodule somewhere inside
   mup.finalize(model)

   # 3. Optimize with a μP-aware optimizer.
   optimizer = mup.MuAdamW(model.parameters(), lr=lr)

That is the whole workflow. Reloading a checkpoint needs **no** extra μP bookkeeping:
reconstruct the model with the same arguments (which reproduces identical base shapes)
and call ``load_state_dict`` as usual.

How it works
------------

The key observation is that, for these backbones, the μP *base shapes* are a
deterministic function of the constructor arguments -- the width is just another
argument (``num_heads`` for the transformers, ``hidden_channels`` for the MLP, the
``hidden_reps`` multiplicities for ``GraphNet``). So when ``parametrization="mup"`` the
backbone re-instantiates itself at a small "base" and "delta" width *in memory* and
computes its base shapes during construction. There are therefore no ``.bsh`` files and
no manually managed base/delta models.

Concretely, ``parametrization="mup"`` changes three things relative to the standard
(``"sp"``) path:

* the readout becomes a :class:`mup.MuReadout` (μP ``1/width`` output scaling),
* attention logits are scaled by ``1/d`` instead of ``1/sqrt(d)``
  (see :class:`~lloca.backbone.attention.LLoCaAttention`),
* the weights get a width-independent μP base initialization, which
  :func:`mup.set_base_shapes` then rescales for the actual width.

The standard-parametrization path is untouched (byte-for-byte), so ``parametrization``
defaults to ``"sp"`` and existing code keeps the exact previous behaviour.

Choosing the base/delta widths
------------------------------

Each backbone has sensible defaults (e.g. ``num_heads`` 2 and 4 for the transformer).
To override them, pass ``mup_base_shapes`` / ``mup_delta_shapes`` -- dictionaries of
constructor-argument overrides:

.. code-block:: python

   net = Transformer(
       ..., num_heads=32, parametrization="mup",
       mup_base_shapes={"num_heads": 4},
       mup_delta_shapes={"num_heads": 8},
   )

The only requirement is that base and delta differ along the width axis; the actual
model width is independent of both.

Validating a setup (coordinate check)
-------------------------------------

The standard way to confirm μP is wired correctly is a *coordinate check*: train at a
range of widths for a few steps and verify the activation magnitudes stay roughly
constant in width (μP) rather than drifting (SP). See
``tests/lloca/mup/test_mup.py::test_coord_check_mlp`` for a minimal, CPU-only example.

.. note::

   μP support for the :class:`~lloca.backbone.particletransformer.ParticleTransformer`
   and :class:`~lloca.backbone.particlenet.ParticleNet` backbones is not provided yet:
   their BatchNorm / LayerScale / token structure needs μP treatment beyond the
   readout/attention/init pattern used here.
