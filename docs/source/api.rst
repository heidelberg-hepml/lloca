API Reference
=============

Equivariant vector prediction
-----------------------------

The Frames-Net uses a small Lorentz-equivariant network to predict a list of vectors. 
We currently have only one option for this ``equivectors`` network, but we plan to add more options in the future.

.. autosummary::
   :toctree: generated/
   :recursive:

   lloca.equivectors.EquiMLP

Frames-Net
----------

The equivariant vectors can be used in a range of Frames-Net procedures to construct local frames.
We support three Lorentz-equivariant Frames-Net approaches, an SO(3)-equivariant approach and an SO(2)-equivariant approach.
In addition, we implement non-equivariant networks as identity frames and data augmentation as random frames.

.. autosummary::
   :toctree: generated/
   :recursive:

   lloca.framesnet.equi_frames.LearnedPDFrames
   lloca.framesnet.equi_frames.LearnedSO13Frames
   lloca.framesnet.equi_frames.LearnedRestFrames
   lloca.framesnet.equi_frames.LearnedSO3Frames
   lloca.framesnet.equi_frames.LearnedSO2Frames
   lloca.framesnet.nonequi_frames.IdentityFrames
   lloca.framesnet.nonequi_frames.RandomFrames

The resulting frames are stored in the ``Frames`` bookkeeping class. 
A range of derived class can be used for efficient access in the backbone architecture.

.. autosummary::
   :toctree: generated/
   :recursive:

   lloca.framesnet.frames.Frames
   lloca.framesnet.frames.InverseFrames
   lloca.framesnet.frames.IndexSelectFrames
   lloca.framesnet.frames.ChangeOfFrames
   lloca.framesnet.frames.LowerIndicesFrames

Backbone networks
-----------------

The LLoCa framework can be used to make generic backbone architectures Lorentz-equivariant.

1. Transform the network inputs into their local frames to make them invariant.
2. For message-passing architectures, transform the messages from the sender frame to the receiver frame using a non-trivial message representation, i.e. not only scalars.
3. Transform the network outputs back to the global frame to obtain a Lorentz-equivariant output. This step is trivial in the case of Lorentz-invariant outputs.

The ``MLP`` does not require any modifications to be used in the LLoCa framework.
For message-passing architectures, we provide the ``LLoCaMessagePassing`` class to conveniently adapt graph networks based on the ``torch_geometric.nn.conv.MessagePassing`` class to the LLoCa framework.
For transformers, we provide the ``LLoCaAttention`` class as a drop-in replacement for ``torch.nn.functional.scaled_dot_product_attention`` and other attention backends.
We demonstrate how to use these tools with a baseline LLoCa-GNN and a LLoCa-Transformer. 
For ``ParticleNet`` and ``ParticleTransformer``, we demonstrate how to use LLoCa with established architectures.

.. autosummary::
    :toctree: generated/
    :recursive:
    
    lloca.backbone.mlp.MLP
    lloca.backbone.lloca_message_passing.LLoCaMessagePassing
    lloca.backbone.attention.LLoCaAttention
    lloca.backbone.graphnet.GraphNet
    lloca.backbone.transformer.Transformer
    lloca.backbone.particlenet.ParticleNet
    lloca.backbone.particletransformer.ParticleTransformer

Lorentz group representations
-----------------------------

The LLoCa framework supports arbitrary Lorentz group representations for inputs, outputs and messages.
The ``TensorReps`` class organizes the properties of these representations,
while the ``TensorRepsTransform`` class implements the actual transformations of features.

.. autosummary::
   :toctree: generated/
   :recursive:

   lloca.reps.tensorreps.TensorReps
   lloca.reps.tensorreps_transform.TensorRepsTransform

Utilities
---------

Finally, we provide a range of utility functions for Lorentz transformations, random transformations and orthogonalization.

.. autosummary::
   :toctree: generated/
   :recursive:

   lloca.utils.utils
   lloca.utils.lorentz
   lloca.utils.rand_transforms
   lloca.utils.orthogonalize_3d
   lloca.utils.orthogonalize_4d
   lloca.utils.polar_decomposition
