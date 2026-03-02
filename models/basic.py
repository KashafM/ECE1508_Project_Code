"""
Basic and Standard U-Net implementations built on the shared BaseUNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from base_model import BaseUNet, DoubleConv, Down, Up, OutConv
from .registry import register_model


class BasicUNet(BaseUNet):
    """Baseline U-Net leveraging the shared encoder/decoder blocks."""

    def __init__(self, n_channels: int = 4, n_classes: int = 5, bilinear: bool = False) -> None:
        super().__init__(n_channels, n_classes, bilinear)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class StandardUNet(BaseUNet):
    """Configurable U-Net with tunable feature width and He init."""

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 5,
        bilinear: bool = False,
        base_features: int = 64,
    ) -> None:
        super().__init__(n_channels, n_classes, bilinear)
        self.base_features = base_features

        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)

        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        self.outc = OutConv(base_features, n_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


register_model(
    name="basic_unet",
    constructor=BasicUNet,
    metadata={
        "display_name": "Basic U-Net",
        "default_args": {"n_channels": 4, "n_classes": 5, "bilinear": False},
        "description": "Baseline four-level U-Net architecture.",
    },
)

register_model(
    name="standard_unet",
    constructor=StandardUNet,
    metadata={
        "display_name": "Standard U-Net",
        "default_args": {"n_channels": 4, "n_classes": 5, "bilinear": False, "base_features": 64},
        "description": "Configurable U-Net with adjustable base feature width and He initialisation.",
    },
)
