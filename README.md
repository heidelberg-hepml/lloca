<div align="center">

## Lorentz Local Canonicalization

[![Tests](https://github.com/heidelberg-hepml/lloca/actions/workflows/tests.yaml/badge.svg)](https://github.com/heidelberg-hepml/lloca/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/heidelberg-hepml/lloca/branch/main/graph/badge.svg)](https://codecov.io/gh/heidelberg-hepml/lloca)
[![PyPI version](https://img.shields.io/pypi/v/lloca.svg)](https://pypi.org/project/lloca)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

[![LLoCa-CS](http://img.shields.io/badge/paper-arxiv.2505.20280-B31B1B.svg)](https://arxiv.org/abs/2505.20280)
[![LLoCa-HEP](http://img.shields.io/badge/paper-arxiv.2508.14898-B31B1B.svg)](https://arxiv.org/abs/2508.14898)

</div>

This repository contains a standalone implementation of **Lorentz Local Canonicalization (LLoCa)** by [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de), [Luigi Favaro](mailto:luigi.favaro@uclouvain.be), Peter Lippmann, Sebastian Pitz, Gerrit Gerhartz, Huilin Qu, Tilman Plehn, and Fred A. Hamprecht. LLoCa uses equivariantly predicted local reference frames and geometric message passing between these frames to make any architecture Lorentz-equivariant.
You can read more about LLoCa in the following two papers and in the [LLoCa documentation](https://heidelberg-hepml.github.io/lloca/):
- [Lorentz Local Canonicalization: How to make any Network Lorentz-Equivariant](https://arxiv.org/abs/2505.20280) (ML audience)
- [Lorentz-Equivariance without Limitations](https://arxiv.org/abs/2508.14898) (HEP audience)

![](img/lloca.png)

## Installation

You can either install the latest release using pip
```
pip install lloca
```
or clone the repository and install the package in dev mode
```
git clone https://github.com/heidelberg-hepml/lloca.git
cd lloca
pip install -e .
```

## How to use LLoCa

Please have a look at the [LLoCa documentation](https://heidelberg-hepml.github.io/lloca/) (WIP) and our example notebook for the [LLoCa-Transformer](examples/demo_transformer.ipynb).

## Features

- Backbone architectures in `lloca/backbone`: `Transformer`, `ParticleTransformer`, `ParticleNet`, `GraphNet`, `MLP`
- The `Transformer` backbone supports several attention kernels that can be installed optionally with e.g. `pip install lloca[xformers_attention]`
- `LLoCaMessagePassing` as blueprint for generic `LLoCa` graph network backbones
- Equivariant vector predictors in `lloca/equivectors`: `EquiMLP`
- Local frames for equivariant architectures on several symmetry groups: $SO(1,3)$ (`LearnedPDFrames`, `LearnedSO13Frames`, `LearnedRestFrames`), $SO(3)$ (`LearnedSO3Frames`) and $SO(2)$ (`LearnedSO2Frames`); as well as the corresponding random global frames for data augmentation
- Support for arbitrary higher-order representations with the `TensorReps` class

Coming soon:

- More `equivectors` options
- Parity-odd representations
- Support for cross-attention

## Examples

- https://github.com/heidelberg-hepml/lorentz-frames: Codebase for the original papers; will eventually be based on the `lloca` repository. For now, development takes place in `lorentz-frames` and `lloca` serves as a stable version for others to work with.

Let us know if you use `lloca`, so we can add your repo to the list!

## Citation

If you find this code useful in your research, please cite our papers

```bibtex
@article{Favaro:2025pgz,
    author = "Favaro, Luigi and Gerhartz, Gerrit and Hamprecht, Fred A. and Lippmann, Peter and Pitz, Sebastian and Plehn, Tilman and Qu, Huilin and Spinner, Jonas",
    title = "{Lorentz-Equivariance without Limitations}",
    eprint = "2508.14898",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "8",
    year = "2025"
}
@article{Spinner:2025prg,
    author = "Spinner, Jonas and Favaro, Luigi and Lippmann, Peter and Pitz, Sebastian and Gerhartz, Gerrit and Plehn, Tilman and Hamprecht, Fred A.",
    title = "{Lorentz Local Canonicalization: How to Make Any Network Lorentz-Equivariant}",
    eprint = "2505.20280",
    archivePrefix = "arXiv",
    primaryClass = "stat.ML",
    month = "5",
    year = "2025"
}
```

