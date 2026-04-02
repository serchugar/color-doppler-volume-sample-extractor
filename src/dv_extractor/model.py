import torch
import torch.nn as nn


class DynamicUNet(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        depth: int = 3,
        init_features: int = 32,
    ) -> None:
        super().__init__()

        self.depth = depth

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
