# Changelog

All notable changes to this project will be documented in this file

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

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

[Unreleased]: https://github.com/serchugar/color-doppler-volume-sample-extractor/compare/v0.1.0...main
[0.1.0]: https://github.com/serchugar/color-doppler-volume-sample-extractor/releases/tag/v0.1.0
