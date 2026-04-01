# Volumetric Segmentation of Color Doppler Samples

This repository implements a U-Net based pipeline to segment and generate masks from volumetric Color Doppler imaging data:

<div style="display:flex; align-items:center; margin: 0 auto; width: fit-content;">
  <img src="assets/img7.jpg" alt="Input Color Doppler Image" width="250">
  <span style="font-size: 24px; margin: 0 20px;">→</span>
  <img src="assets/mask7.png" alt="Binary Mask Output Segmentation Mask" width="250">
</div>

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and enter the repo
git clone https://github.com/serchugar/color-doppler-volume-sample-extractor
cd color-doppler-volume-sample-extractor

# Sync environment and dependencies. Choose one of the two
uv sync --extra cuda # if nvidia gpu
uv sync --extra cpu # else

# Prepare your data (see training workflow below)

# Copy and set the .env file
copy .env.example .env # if windows
cp .env.example .env # if linux

# Run the main script. Modify training params here
uv run main.py

# Weights of the trained model will be stored in:
/weights
```

## Training Workflow

To train the U-Net model, follow these steps:

### 1. Prepare Your Data
Organize your training images and masks in a directory with the following naming convention:
- Images: `img<number>.jpg` (e.g., `img1.jpg`, `img2.jpg`, `img123.jpg`)
- Masks: `mask<number>.png` (e.g., `mask1.png`, `mask2.png`, `mask123.png`)

**Important:** The number in the filename must match between corresponding image and mask files (e.g., `img42.jpg` should have a corresponding `mask42.png`). This naming convention is required for the pipeline to correctly associate images with their segmentation masks.

> Mask images must be **binary images without antialiasing** and saved as PNG files to preserve lossless compression.

### 2. Configure Environment Variables
Create a `.env` file in the root directory of the project and add the path to your labeled data:

```
LABELED_DATA_DIR="/path/to/your/data/directory"
```

Replace `/path/to/your/data/directory` with the absolute or relative path to the folder containing your `img<number>.jpg` and `mask<number>.png` files.

Example:
```
LABELED_DATA_DIR="./data/labeled"
```

### 3. Run Training
Execute the main script:

```bash
uv run main.py
```
Any changes to the training hyperparameters can be done in the `train()` function call within `main.py`.

The model will load the images and masks from the directory specified in `LABELED_DATA_DIR` and begin training.
If `checkpoint_dir` in `main.py` is enabled (it is by default), it will save the best model to a `/weights` folder.

> **Note:** Due to the lack of data, and the time cost of creating each mask, the training does not run a validation
process.

## GPU Acceleration
To enable CUDA support for faster training and inference, ensure you have CUDA and CUDNN installed.  

In order make opencv work with cuda, copy all the binaries from CUDNN into the bin folder of CUDA, normally located at:  

`C:\Program Files\NVIDIA\CUDNN\v9.20\bin`  
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin`  

Then, sync the environment with the `cuda` extra:

```bash
uv sync --extra cuda
```