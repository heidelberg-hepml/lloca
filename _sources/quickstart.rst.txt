Quickstart
==========

This page sets you up for building and running LLoCa networks.

Installation
------------

Before using the package, install it via pip:

.. code-block:: bash

   pip install lloca

Alternatively, if you're developing locally:

.. code-block:: bash

   git clone https://github.com/heidelberg-hepml/lloca.git
   cd lloca
   pip install -e .

Building a LLoCa-Transformer
----------------------------

.. image:: /_static/lloca.png
   :align: center
   :width: 100%

We now demonstrate how to build a LLoCa-Transformer, the most efficient architecture based on our papers. 
We build the LLoCa-Transformer in three steps, following the picture above:

1. Construct local frames based on three equivariantly predicted vectors
2. Transform particle features into local frames
3. Process local particle features with any backbone architecture

0. Generate particle data
-------------------------

We start by generating toy particle data, for instance for an amplitude regression task. 
We describe particles by a four-momentum and one scalar feature, for instance the particle type. 
Using random numbers, we generate a batch of 128 events with 10 particles each.

.. code-block:: python

   import torch
   num_scalars = 1
   B, N = 128, 10
   mass = 1
   p3 = torch.randn(B, N, 3)
   E = (mass**2 + (p3**2).sum(dim=-1, keepdims=True)).sqrt()
   fourmomenta = torch.cat([E, p3], dim=-1) # (128, 10, 4)
   scalars = torch.randn(B, N, num_scalars) # (128, 10, 1)

1. Construct local frames based on three equivariantly predicted vectors
--------------------------------------------------------------------

Given these particle features, we want to construct a local frame :math:`L` for each particle. 
The local frames are Lorentz transformations, i.e. they satisfy :math:`L^TgL=g` with :math:`L\in \mathbb{R}^{4\times 4}`. 
We further design them to satisfy the transformation behavior :math:`L\overset{\Lambda}{\to} L\Lambda^{-1}` under Lorentz transformations :math:`\Lambda`, 
this ensures that particle features in the local frame are invariant.

We construct the local frames in two steps. First, we use a simple Lorentz-equivariant ``equivectors`` network, :mod:`~lloca.equivectors.EquiMLP`, to construct 3 vectors.

.. code-block:: python

   from lloca.equivectors.equimlp import EquiMLP

   def equivectors_constructor(n_vectors):
      return EquiMLP(
         n_vectors=n_vectors,
         num_blocks=2,
         num_scalars=num_scalars,
         hidden_channels=8,
         num_layers_mlp=2,
      )

   # quickly test it
   equivectors_test = equivectors_constructor(3)
   vectors = equivectors_test(fourmomenta, scalars) # (128, 10, 3, 4)

Next, we define the ``framesnet`` class which calls the ``equivectors`` to predict a set of vectors
and further performs the orthonormalization to construct the local ``frames``.
In our minimal example, we use the :mod:`~lloca.framesnet.equi_frames.LearnedPDFrames` ``framesnet`` and
we pass the constructor as :mod:`equivectors=equivectors_constructor`.

.. code-block:: python

   from lloca.framesnet.equi_frames import LearnedPDFrames

   framesnet = LearnedPDFrames(equivectors=equivectors_constructor)
   frames = framesnet(fourmomenta, scalars) # (128, 10, 4, 4)

The package implements many alternative ``framesnet`` choices:

- :mod:`~lloca.framesnet.equi_frames.LearnedPDFrames`: Construct a learned Lorentz transformation from a boost and a rotation, i.e. following a polar decomposition, with the rotation constructed using the Gram-Schmidt algorithm in the 3-dimensional euclidean space. This is the default Lorentz-equivariant ``framesnet``.
- :mod:`~lloca.framesnet.equi_frames.LearnedSO13Frames`: Construct a learned Lorentz transformation directly using the Gram-Schmidt algorithm in Minkowski space. The result is equivalent to ``LearnedPD``, but ``LearnedPD`` has the advantage of providing direct access to the boost, which is useful in some cases.
- :mod:`~lloca.framesnet.equi_frames.LearnedSO3Frames` and :mod:`~lloca.framesnet.equi_frames.LearnedSO2Frames`: Construct learned :math:`SO(2)` and :math:`SO(3)` transformations, embedded in the Lorentz group. The resulting architectures are :math:`SO(2)`- and :math:`SO(3)`-equivariant, respectively.
- :mod:`~lloca.framesnet.nonequi_frames.RandomFrames`: Random global frames, corresponding to data augmentation.
- :mod:`~lloca.framesnet.nonequi_frames.IdentityFrames`: Frames from identity transforms, corresponding to the baseline non-equivariant architectures.

2. Transform particle features into local frames
------------------------------------------------

Once the frames are constructed, we have to transform the particle features into their local frames. 
We use the local frames transformation for the four-momenta, whereas the scalar features are already invariant by definition.

.. code-block:: python

   from lloca.reps.tensorreps_transform import TensorReps, TensorRepsTransform

   fourmomenta_rep = TensorReps("1x1n")
   trafo_fourmomenta = TensorRepsTransform(fourmomenta_rep)
   fourmomenta_local = trafo_fourmomenta(fourmomenta, frames) # (128, 10, 4)

   features_local = torch.cat([fourmomenta_local, scalars], dim=-1) # (128, 10, 5)

The ``lloca`` package implements arbitrary Lorentz tensors through the :mod:`~lloca.reps.tensorreps.TensorReps` class, 
and their transformation behavior with :mod:`~lloca.reps.tensorreps_transform.TensorRepsTransform`. 
We denote ``0n`` for scalar, ``1n`` for vector, ``2n`` for rank 2 tensor, and so on, 
where the ``n`` stands for *normal* in contrast to *parity-odd* (not fully supported). 
General representations can be obtained by linear combinations of these fundamentals, e.g. ``4x0n+8x1n+3x2n+2x3n``.

3. Process local particle features with any backbone architecture
-----------------------------------------------------------------

Given the particle features in the local frame, we can process them with any backbone architecture without violating Lorentz-equivariance. 
To obtain an equivariant prediction, we have to finally transform the output features from the local into the global frames, 
however this step is trivial if the output features are scalar.

There is one caveat regarding the backbone architecture: 
To allow a meaningful message-passing, we have to properly transform particle features when they are communicated between particles. 
This manifests in a modification of the attention mechanism for transformers, and in the message-passing for graph networks. 
This aspect is already implemented in the backbones available in ``lloca/backbone/``, and has to be added for new backbone architectures within LLoCa.
This is already handled internally for the LLoCa :mod:`~lloca.backbone.transformer.Transformer` and other architectures in ``lloca/backbone/``

.. code-block:: python

   from lloca.backbone.transformer import Transformer

   backbone = Transformer(
      in_channels=4+num_scalars,
      attn_reps="4x0n+1x1n",
      out_channels=1,
      num_blocks=2,
      num_heads=2,
   )

   out = backbone(features_local, frames) # (128, 10, 1)
   
