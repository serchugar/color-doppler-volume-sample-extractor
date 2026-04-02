from collections.abc import Sequence
from pathlib import Path

import cv2 as cv
import torch
import torchvision.transforms.v2 as v2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform


def discover_images(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("img*.jpg"))


def discover_images_with_mask(data_dir: Path) -> tuple[list[Path], list[Path]]:
    images = sorted(data_dir.rglob("img*.jpg"))
    masks = sorted(data_dir.rglob("mask*.png"))
    max_len = min(len(images), len(masks))
    return images[:max_len], masks[:max_len]


class DopplerDataset(Dataset):
    def __init__(
        self,
        images: list[Path],
        masks: list[Path] | None = None,
        size: Sequence[int] = (512, 512),
        transform: Transform | None = None,
        threshold: float = 0.95,
    ) -> None:
        super().__init__()
        self.images = images
        self.masks = masks
        self.size = list(size)
        self.transform = transform
        self.threshold = threshold

        self._image_cache = {}
        self._mask_cache = {}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | Tensor:
        image = cv.imread(self.images[idx], cv.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image at path '{self.images[idx]}'")
        _, image = cv.threshold(image, 255 * self.threshold, 255, cv.THRESH_BINARY)
        image = tv_tensors.Mask(torch.from_numpy(image).unsqueeze(0))  # [1,H,W]

        mask = None
        if self.masks is not None:
            mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask at path '{self.masks[idx]}'")
            mask = tv_tensors.Mask(torch.from_numpy(mask).unsqueeze(0))  # [1,H,W]

        if self.transform:
            if mask is not None:
                image, mask = self.transform(image, mask)
            else:
                image = self.transform(image)

        image = v2.functional.resize(image, self.size, antialias=False)
        image = (image > 127.5).float()

        if mask is not None:
            mask = v2.functional.resize(mask, self.size, antialias=False)
            mask = (mask > 127.5).float()
            return image, mask

        return image
