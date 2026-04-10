from pathlib import Path
from typing import overload

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as f
from torchvision.io import decode_image


class DynamicUNet(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        init_features: int = 32,
        threshold: float = 0.95,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.threshold = threshold

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        curr_channels = in_channels
        features = init_features

        # Encoder
        for _ in range(depth):
            self.encoders.append(self._block(curr_channels, features))
            self.pools.append(nn.MaxPool2d(2))
            curr_channels = features
            features *= 2

        self.bottleneck = self._block(curr_channels, features)

        # Decoder
        for _ in range(depth):
            curr_channels = features // 2
            self.upconvs.append(nn.ConvTranspose2d(features, curr_channels, 2, 2))
            self.decoders.append(self._block(features, curr_channels))
            features = curr_channels

        self.final_conv = nn.Conv2d(init_features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for i in range(self.depth):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            x = torch.cat((x, skip), dim=1)
            x = self.decoders[i](x)

        # Don't return torch.sigmoid(self.final_conv(x))
        # Sigmoid will be implicitly applied with the loss function nn.BCEWithLogitsLoss()
        return self.final_conv(x)

    @staticmethod
    def _block(in_c: int, out_c: int, dropout_prob: float = 0.1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @overload
    def predict(self, img: Path | torch.Tensor) -> torch.Tensor: ...

    @overload
    def predict(self, img: list[Path] | list[torch.Tensor]) -> list[torch.Tensor]: ...

    def predict(self, img):
        from tqdm import tqdm

        self.eval()
        custom_format = " {desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed_s:0.1f}s elapsed, {remaining_s:0.0f}s remaining, {rate_fmt}]         "  # noqa: E501

        if isinstance(img, Path):
            return self._predict_one(decode_image(img))

        elif isinstance(img, torch.Tensor):
            return self._predict_one(img)

        elif isinstance(img, list):
            print("Computing inference...")
            if isinstance(img[0], Path):
                masks = []
                for img_path in tqdm(img, bar_format=custom_format, unit=" samples"):
                    mask = self._predict_one(decode_image(img_path))
                    masks.append(mask)
                return masks

            elif isinstance(img[0], torch.Tensor):
                return [self._predict_one(i) for i in tqdm(img, bar_format=custom_format, unit=" samples")]
        else:
            raise ValueError(f"Invalid input type: {type(img)}")

    def _predict_one(self, img: torch.Tensor) -> torch.Tensor:
        img = f.to_dtype(img, torch.float32)
        img = (f.to_grayscale(img) > 255 * self.threshold).float()
        img = img.unsqueeze(0)
        img = img.to(self.device)

        _, _, h, w = img.shape
        img = v2.Resize((512, 512))(img)  # Resize to 512x512 since model was trained on 512x512 images

        with torch.no_grad():
            probs = torch.sigmoid(self.forward(img))
            probs = v2.Resize((h, w), interpolation=v2.InterpolationMode.NEAREST)(probs)
            return ((probs > 0.5).float()).squeeze()  # Remove batch dimension

    def load_weights(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
