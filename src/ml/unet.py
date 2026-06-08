"""Lightweight U-Net for BEV segmentation.

Input schema:
    Tensor shape ``(B, 1, H, W)`` with per-image normalized BEV intensity.

Output schema:
    Tensor shape ``(B, C, H, W)`` with per-class logits.

Coordinate frames:
    BEV tensors are rasterized from world-frame ground points. The model
    preserves pixel height and width so output masks stay aligned with the
    input BEV world-grid metadata.
"""

from __future__ import annotations

import torch
from torch import nn


CONV_KERNEL = 3
POOL_SCALE = 2
HEAD_KERNEL = 1


class ConvBNReLU(nn.Module):
    """Two Conv2d + BatchNorm + ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        padding = CONV_KERNEL // POOL_SCALE
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                CONV_KERNEL,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                CONV_KERNEL,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


class BEVSegNet(nn.Module):
    """U-Net for BEV lane and road-marking segmentation."""

    def __init__(self, num_classes: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBNReLU(1, base_channels)
        self.enc2 = ConvBNReLU(base_channels, base_channels * 2)
        self.enc3 = ConvBNReLU(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBNReLU(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(POOL_SCALE)
        self.bottleneck = ConvBNReLU(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            POOL_SCALE,
            stride=POOL_SCALE,
        )
        self.dec4 = ConvBNReLU(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            POOL_SCALE,
            stride=POOL_SCALE,
        )
        self.dec3 = ConvBNReLU(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            POOL_SCALE,
            stride=POOL_SCALE,
        )
        self.dec2 = ConvBNReLU(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            POOL_SCALE,
            stride=POOL_SCALE,
        )
        self.dec1 = ConvBNReLU(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, HEAD_KERNEL)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run shape-preserving BEV segmentation.

        Args:
            tensor: ``(B, 1, H, W)`` BEV image. FRAME: world-grid raster.

        Returns:
            ``(B, C, H, W)`` logits. FRAME: same BEV world-grid raster.
        """
        enc1 = self.enc1(tensor)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.head(dec1)
