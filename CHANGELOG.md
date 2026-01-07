# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.4] - 07.01.2026

### Added

- FlashAttention varlen attention backend https://github.com/Dao-AILab/flash-attention

### Changed

- Collect optional requirements in `[dev]` extra

## [1.3.3] - 19.12.2025

### Added

- Complete documentation (was very preliminary before)

### Changed

- Code style improvements in `utils/rand_transform.py`
- Update demo notebook
- Improve Frames-Net docs formatting
- Deprecated the `deterministic_boost` option in `LearnedPDFrames`

## [1.3.2] - 21.11.2025

### Added

- `use_amp` option for `equivectors`

### Changed

- Make `transformer.py` channels definitions more intuitive (thanks canisli)
- Fully move to ruff as the formatter
- Make `LLoCaAttention` more modular

### Fixed

- Show correct pip-install-with-extras commands, e.g. `pip install lloca[xformers_attention]` -> `pip install lloca[xformers-attention]` (pypi doesn't support `_`)

### Removed

- `requirements.txt` (already in `pyproject.toml`)
- `get_xformers_attention_mask` function (should be manually defined in experiment code)

## [1.3.1] - 10.11.202

_Fix import bugs._

## [1.3.0] - 10.11.2025

### Added

- `LGATrVectors` and `PELICANVectors` as alternatives to `MLPVectors`
- `ruff` formatting
- `CHANGELOG.md`

### Changed

- Rename `equivectors.equimlp.EquiMLP` -> `equivectors.mlp.EquiMLP`
- Turn `checks` in orthonormalization off by default (avoids GPU/CPU sync)
- Review regularization
  - Apply `reg_lightlike` also in `polar_decomposition`
  - Ensure that the final vector is timelike in `reg_lightlike` (constrain noise to have positive components)
  - Set `eps=None` by default, corresponding to `torch.finfo(dtype).eps`

## [1.2.0] - 27.10.2025

### Added

- `compile` option for `framesnet` orthonormalization
- `LearnedZFrames`

### Changed

- Overall many efficiency rewrites
- Add `init_standardization()` method to `EquiVectors` instead of initializing in the first iteration

## [1.1.0] - 03.10.2025

### Added

- `docs/` with gh-pages (still has several 'Coming soon' pages)

### Changed

- Update defaults in `EquiMLP`: `num_blocks=1`, `hidden_vectors=1`, use the most stable setting as default
- Unify naming convention for `get_edge_index_...` functions and add unit tests for them
- Small changes in `transformer.py` to improve readability
- Unify docstrings

### Fixed

- Bug in attention backend selection

## [1.0.1] - 15.09.2025

### Added

- `examples/demo_transformer.ipynb`
- `__init__.py` files

## [1.0.0] - 13.10.2025

_First release._
