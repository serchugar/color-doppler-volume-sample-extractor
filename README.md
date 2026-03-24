# Volumetric Segmentation of Color Doppler Samples

This repository implements a U-Net based pipeline to segment and generate masks from volumetric Color Doppler imaging data.

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and enter the repo
git clone https://github.com/serchugar/color-doppler-volume-sample-extractor
cd color-doppler-volume-sample-extractor

# Sync environment and dependencies. Choose one of the two.
uv sync --extra cuda # if nvidia gpu
uv sync --extra cpu # else

# Run the main script
uv run main.py
```

## GPU Acceleration
To enable CUDA support for faster training and inference, ensure you have CUDA and CUDNN installed.  

In order make opencv work with cuda, copy all the binaries from CUDNN into the bin folder of CUDA, normally located at:  

`C:\Program Files\NVIDIA\CUDNN\v9.20\bin`  
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin`  

Then, sync the environment with the `cuda` extra:

```bash
uv sync --extra cuda
```