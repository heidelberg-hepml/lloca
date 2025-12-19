LLoCa vs L-GATr
===============

Most previously published Lorentz-equivariant networks rely on specialized layers to achieve Lorentz-equivariance. Popular examples are

* `LorentzNet <https://arxiv.org/abs/2201.08187>`_, a Lorentz-equivariant graph network using scalar and vector representations
* `PELICAN <https://arxiv.org/abs/2307.16506>`_, a Lorentz-invariant graph network that uses general permutation-equivariant layers operating on Lorentz invariants
* `CGENN <https://arxiv.org/abs/2305.11141>`_, a Lorentz-equivariant graph network using geometric algebra representations
* `L-GATr <https://arxiv.org/abs/2411.00446>`_, a Lorentz-equivariant transformer using geometric algebra representations

Canonicalization, or specifically Lorentz local canonicalization (LLoCa), is an alternative approach that extends an existing backbone architecture with a
Lorentz-equivariant canonicalization procedure to achieve Lorentz-equivariance. This section discusses the benefits of both approaches and helps to decide
which one to use for your specific application. We focus on L-GATr as a representative of the specialized-layer approach because we compared it carefully with
LLoCa-Transformers, but most arguments apply to other specialized-layer networks as well.

**Disclaimer:** Lorentz-equivariant networks are great, but when starting on a task from scratch we recommend to start with a non-equivariant baseline network
first, for instance a vanilla transformer, and move to a Lorentz-equivariant network later on for the extra performance and robustness.

Benefits of L-GATr / specialized layers
---------------------------------------

* **Simpler**, e.g. `LGATrSlim <https://github.com/heidelberg-hepml/lgatr/blob/main/lgatr/nets/lgatr_slim.py>`_ fits into a single file.
  Canonicalization modifies the backbone architecture only slightly, but the Frames-Net and subsequent orthonormalization requires significant extra code.
* **Training dynamics are typically easier**, because the interaction between the Frames-Net and backbone in LLoCa can lead to more complex training dynamics.
  For instance, we found in some cases that the Frames-Net in LLoCa overfits before the backbone does, even though the Frames-Net has far fewer parameters.

Benefits of LLoCa / canonicalization
------------------------------------

* **Make any backbone architecture Lorentz-equivariant**: We constructed LLoCa-Transformer, LLoCa-GNN, LLoCa-ParticleNet and LLoCa-ParT networks in our publications.
  LLoCa is particularly useful when your non-equivariant architecture already includes many tricks that would be time-consuming to re-implement in a
  Lorentz-equivariance-by-layers architecture like L-GATr, such as dynamic graph convolutions, attention mechanisms, or U-Net layouts.
  See :doc:`more-backbones/index` for instructions and code examples for how to extend your favorite architecture with LLoCa.
* **Higher-order representations** are not only straight-forward to include in LLoCa but also easily mixed in equal or unequal fractions. LLoCa builds on generic tensor representations of the Lorentz group through the
  :class:`~lloca.reps.tensorreps.TensorReps` class, e.g. ``16x0n+8x1n+2x2n+1x3n`` for a direct product of 16 scalar, 8 vector, 2 second-rank tensor, and 1 third-rank tensor representations.
  Arbitrary higher-order representations are implemented, but using them comes at the cost of slower inference because of the additional matrix multiplications.
* **Reduced resource constraints**: For similar network size, we find that LLoCa networks typically require less GPU memory and FLOPs than comparable specialized-layer networks.
  This is because the backbone processes Lorentz-invariant features only, which are typically lower-dimensional and cheaper to process.
  The overhead from the Frames-Net and frame-to-frame conversions in message passing increases inference time by a factor of 1.5-2x in our current implementation,
  but based on our FLOPs analyses we believe that there is significant room for optimization that we want to explore in the future.
* **Flexible for studies on subgroup equivariance and data augmentation**: Subgroup-equivariant networks such as SO(3)-equivariant or SO(2)-equivariant networks as well as data augmentation
  can be implemented easily as modifications in the Frames-Net. This allows for systematic studies of the impact of symmetry-aware designs, which would require entire new network designs
  when using the specialized-layer approach.
