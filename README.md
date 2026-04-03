# Volumetric Segmentation of Color Doppler Samples

This repository implements a U-Net based pipeline to segment and generate masks from volumetric Color Doppler imaging data:

<div align="center">
  <img src="assets/img7.jpg" alt="Input Color Doppler Image" width="250">
  <span>→</span>
  <img src="assets/mask7.png" alt="Binary Mask Output Segmentation Mask" width="250">
</div>

> [!NOTE]
> To see the latest updates and upcoming features, please check the [Change Log](./CHANGELOG.md).

## Quick Start
For [uv](https://docs.astral.sh/uv/) users (recommended):

```bash
git clone https://github.com/serchugar/color-doppler-volume-sample-extractor

# In your uv project, install in editable mode with CUDA support (Nvidia GPU)
uv pip install -e "/path/to/color-doppler-volume-sample-extractor[cuda]"

# Or install in editable mode with CPU support
uv pip install -e "/path/to/color-doppler-volume-sample-extractor[cpu]"
```

Alternatively, if using standard Python and pip:

```bash
git clone https://github.com/serchugar/color-doppler-volume-sample-extractor

# Install with CUDA support
pip install -e "/path/to/color-doppler-volume-sample-extractor[cuda]"

# Or install with CPU support
pip install -e "/path/to/color-doppler-volume-sample-extractor[cpu]"
```

## Training Workflow
If you want to skip training and use a pre-trained model, you can download the weights from the [latest release](https://github.com/serchugar/color-doppler-volume-sample-extractor/releases/latest).

### 1. Prepare Your Data
Organize your training images and masks in a directory with the following naming convention:
- Images: `img<number>.jpg` (e.g., `img1.jpg`, `img2.jpg`, `img123.jpg`)
- Masks: `mask<number>.png` (e.g., `mask1.png`, `mask2.png`, `mask123.png`)

> [!IMPORTANT]
> The number in the filename must match between corresponding image and mask files (e.g., `img42.jpg` should have a corresponding `mask42.png`). This naming convention is required for the pipeline to correctly associate images with their segmentation masks.
>
> Mask images must be **binary images without antialiasing** and saved as PNG files to preserve lossless compression.

### 2. Run Training
```python
import random
from pathlib import Path

from dv_extractor import DEVICE, DynamicUNet, train
from dv_extractor.utils import seed_all

# Not mandatory, but recommended for reproducibility
seed = random.getrandbits(32)
seed_all(seed)
print(f"Seed: {seed}")

model = DynamicUNet(in_channels=1, out_channels=1, depth=4, init_features=32)
model.to(DEVICE)
print(f"Model device: {model.device}\n")

labeled_data_dir = Path("path/to/your/labeled/data/dir")
train(
    model,
    labeled_data_dir,
    epochs=100,
    lr=0.001,
    batch_size=5,
    checkpoints_dir=Path("weights"),
)
```

The trained model weights will be saved in the `checkpoints_dir` folder.

> [!NOTE]
> Due to the lack of data, and the time cost of creating each mask, the training does not run a validation process.

## Inference
To run predictions, first train your model or load pretrained weights, then use the `predict()` method from the `DynamicUNet` class.

> [!WARNING]
> The current pretrained weights were trained on images after applying a 95% threshold.  
> Loading "raw" images directly into the model without this thresholding will result in incorrect segmentations.  
> The `predict()` method handles this automatically, use it to avoid any issues. 

```python
from pathlib import Path

import torch
from dv_extractor import DEVICE, DynamicUNet, discover_images
from dv_extractor.utils import visualize
from torchvision.io import decode_image

# Initialize model with the correct hyperparameters
model = DynamicUNet(in_channels=1, out_channels=1, depth=4, init_features=32)
model.to(DEVICE)

# Load the weights. Here we use the pretrained ones
weights_path = Path("weights/pretrained/unet_depth4_feat32_in1_out1_weights.pt")
state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

# Load the images 
imgs_path: list[Path] = discover_images(Path("path/to/your/images"))

# Run the inference
masks: list[torch.Tensor] = model.predict(imgs_path)

# Sample image and mask and show on screen
img: torch.Tensor = decode_image(imgs_path[0])
mask: torch.Tensor = masks[0]
visualize(img)
visualize(mask)
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