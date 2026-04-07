# Changelog

All notable changes to this project will be documented in this file

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

### Added
- `predict` method in `DynamicUNet` class. Internally applies threshold. Recommended method for inference.
- `tqdm` dependency for progress bars. Used in `predict` method
- `visualize` function in `utils.py` for visualizing single tensor images
- Estimated remaining time in `train` function epoch logs
- `sub_min_precision` parameter in `format_time` utility function for higher resolution on short durations
- `fiftyone` dependency
- `visualize_predictions` function in `utils.py` for visualizing inference results with the `fiftyone` web app. Allows to visualize thousands of images fast and displays masks as overlays on top of the original images. Metadata can be optionally computed with a bool param in the function signature. If the param 'persist' is set to True, closing the app from the browser will not stop the FiftyOne web app process, it will only finish if the python program itself finishes. This function is exported in `__init__.py`
- `threshold` parameter in `DynamicUNet` constructor to set the threshold applied to the input images before feeding them to the model. Default is 0.95, which is the threshold used for training the pretrained weights. This threshold is propagated within the `train` function to the creation of `DopplerDataset` instances, so it is applied during training as well. The `predict` method also takes into account this threshold
- "Computing inference..." print message in `predict` method for when input is a list. So user knows what `tqdm` progress bar is referring to

### Changed
- Discovery functions in `dataset.py` now support custom regex patterns and multiple extensions (previously restricted to hardcoded patterns)
- `train` function now prints every epoch and highlights new best performance with a "NEW BEST" flag
- `DopplerDataset` `__getitem__` no longer uses OpenCV to load image and apply threshold. Done with pytorch instead.

### Removed
- `opencv-contrib-python` dependency

## [0.2.0] - 2026-04-02

### Added
- `constants.py` module exporting `DEVICE` for device management

### Changed
- Restructured project from App to Python Package with `src/` layout
- Package now installable and importable as `dv_extractor`
- Updated documentation for package usage patterns

### Removed
- `main.py` entry point
- `config/` module (environment management now user-responsibility)

## [0.1.0] - 2026-03-28

### Added
- Initial project scaffold with a `main.py` entry point
- Project metadata and dependency management with `uv`
- Optional dependency groups for `cpu` and `cuda`
- CUDA package sources in `pyproject.toml` for PyTorch and OpenCV
- Quick-start and GPU setup documentation in `README.md`
- MIT `LICENSE` file
- Configuration module with environment-based settings (`Consts` and `Secrets`)
- Environment variable management with `.env` file support and validation
- `.env.example` template for configuration setup
- `Ruff` linter and code formatter with comprehensive configuration
- U-Net `neural network model` implementation for volume extraction
- `Training pipeline` with checkpoint management
- `Dataset` loading utilities for labeled and unlabeled data
- Utility functions for `reproducibility` and device setup

[Unreleased]: https://github.com/serchugar/color-doppler-volume-sample-extractor/compare/v0.2.0...main
[0.2.0]: https://github.com/serchugar/color-doppler-volume-sample-extractor/releases/tag/v0.2.0
[0.1.0]: https://github.com/serchugar/color-doppler-volume-sample-extractor/releases/tag/v0.1.0
