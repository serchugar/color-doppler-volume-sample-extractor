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

### 4. Inference

To run predictions, it is necessary to load the model weights.  
After training the model, a `weights.pt` file will be generated inside /weights.

If you want to skip training and use a pre-trained model, you can download the weights from the [latest release](https://github.com/serchugar/color-doppler-volume-sample-extractor/releases/latest).

> **IMPORTANT: Data Pre-processing Warning**  
> The current pretrained weights were trained on images after applying a 95% threshold.  
> To get accurate predictions, input images must be processed using the DopplerDataset class, which handles this transformation.  
> Loading "raw" images directly into the model without this thresholding will result in incorrect segmentations.

```python
from pathlib import Path

import torch
import torchvision.transforms as T

from config import Consts, Secrets
from src.model import DynamicUNet
from src.dataset import DopplerDataset, discover_images

# Initialize model with the correct hyperparameters
model = DynamicUNet(depth=4, init_features=32, in_channels=1, out_channels=1)
model.to(Consts.DEVICE)

# Load the weights
weights_path = "weights/pretrained/unet_depth4_feat32_in1_out1_weights.pt"
state_dict = torch.load(weights_path, map_location=Consts.DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Load the images with the dataset
images_path: list[Path] = discover_images(Secrets.UNLABELED_DATA_DIR)
dataset: DopplerDataset = DopplerDataset(images_path)

# Sample one image and add batch dimension
image: torch.Tensor = dataset[0]  # type:ignore
image = image.unsqueeze(0).to(Consts.DEVICE)

# Inference
with torch.no_grad():
  output_mask: torch.Tensor = (torch.sigmoid(model(image)) > 0.5).float()

# Show result on screen
mask_img = T.ToPILImage()(output_mask.squeeze().cpu())
mask_img.show()
```

### Pretrained Model Configuration
The following hyperparameters were used to generate the weights available in the [latest release](https://github.com/serchugar/color-doppler-volume-sample-extractor/releases/latest). You can use these as a reference for your own training:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Architecture** | U-Net | Base model structure |
| **Depth** | 4 | Number of downsampling/upsampling blocks |
| **Init Features** | 32 | Number of filters in the first layer |
| **Input Size** | 512 x 512 | Spatial resolution of the samples |
| **Threshold** | 0.95 | Doppler intensity cutoff during preprocessing |
| **Learning Rate** | 0.001 | Optimizer step size (Adam) |
| **Epochs** | 2000 | Total training iterations |
| **Batch Size** | 5 | Number of samples per training step |

## GPU Acceleration
To enable CUDA support for faster training and inference, ensure you have CUDA and CUDNN installed.  

In order [make opencv work with cuda in Windows](https://github.com/cudawarped/opencv-python-cuda-wheels/releases/tag/4.13.0.90), copy all the binaries from CUDNN into the bin folder of CUDA, normally located at:  

`C:\Program Files\NVIDIA\CUDNN\v9.20\bin`  
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin`  

Then, sync the environment with the `cuda` extra:

```bash
uv sync --extra cuda
```

### Troubleshooting: ImportError with cv2

If you encounter an error such as:
```
File "your_dir\color-doppler-volume-sample-extractor\.venv\Lib\site-packages\cv2\__init__.py", line 181, in <module>
  bootstrap()
  ~~~~~~~~~^^
File "your_dir\color-doppler-volume-sample-extractor\.venv\Lib\site-packages\cv2\__init__.py", line 153, in bootstrap
  native_module = importlib.import_module("cv2")
File "C:\Users\your_user\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none\Lib\importlib\__init__.py", line 88, in import_module
  return _bootstrap._gcd_import(name[level:], package, level)

ImportError: DLL load failed while importing cv2: The specified module could not be found.
```

This indicates that the CUDNN DLL files are not accessible to OpenCV.  
To fix this, ensure you have completed the step above in [GPU Acceleration](#gpu-acceleration).