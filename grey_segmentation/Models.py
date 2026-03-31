"""
Models.py
---------
Six UNet variants for brain tumour segmentation:

  2D models — input [B, 1, H, W]       → output [B, NC, H, W]
    UNet2DSmall    (16→32→64,  bottleneck 128)
    UNet2DMedium   (32→64→128→256, bottleneck 512)
    UNet2DDropout  (same as Medium + Dropout2d)

  3D models — input [B, 1, 4, H, W]    → output [B, NC, 4, H, W]
    UNet3DSmall    (16→32→64,  bottleneck 128)
    UNet3DMedium   (32→64→128→256, bottleneck 512)
    UNet3DDropout  (same as Medium + Dropout3d)

  3D pooling uses MaxPool3d((1,2,2)) to preserve the 4-slice depth
  dimension throughout. The double_conv_3d block uses a (1,3,3) kernel
  first (per-slice spatial features) then (3,3,3) (inter-slice context).
"""

import torch
import torch.nn as nn

from DataGen import NUM_CLASSES


# ──────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────

def double_conv_2d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def double_conv_3d(in_ch, out_ch):
    """
    Two-stage 3D conv:
      (1,3,3) — process each slice independently (spatial features)
      (3,3,3) — fuse across slices (inter-slice context)
    Both preserve spatial and depth dimensions via padding.
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, (1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, (3, 3, 3), padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


# ──────────────────────────────────────────────────────────
# 2D UNets
# ──────────────────────────────────────────────────────────

class UNet2DSmall(nn.Module):
    """Shallow 2D UNet. Filters: 16→32→64, bottleneck 128."""

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = double_conv_2d(in_channels, 16)
        self.enc2 = double_conv_2d(16, 32)
        self.enc3 = double_conv_2d(32, 64)
        self.bottleneck = double_conv_2d(64, 128)
        self.up3  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = double_conv_2d(128, 64)
        self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv_2d(64, 32)
        self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv_2d(32, 16)
        self.out  = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class UNet2DMedium(nn.Module):
    """Standard 2D UNet. Filters: 32→64→128→256, bottleneck 512."""

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = double_conv_2d(in_channels, 32)
        self.enc2 = double_conv_2d(32, 64)
        self.enc3 = double_conv_2d(64, 128)
        self.enc4 = double_conv_2d(128, 256)
        self.bottleneck = double_conv_2d(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = double_conv_2d(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv_2d(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = double_conv_2d(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = double_conv_2d(64, 32)
        self.out  = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class UNet2DDropout(nn.Module):
    """2D UNet + Dropout2d. Filters: 32→64→128→256, bottleneck 512."""

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        self.enc1 = double_conv_2d(in_channels, 32)
        self.enc2 = double_conv_2d(32, 64)
        self.enc3 = double_conv_2d(64, 128)
        self.enc4 = double_conv_2d(128, 256)
        self.bottleneck = double_conv_2d(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = double_conv_2d(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv_2d(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = double_conv_2d(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = double_conv_2d(64, 32)
        self.out  = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.drop(self.enc3(self.pool(e2)))
        e4 = self.drop(self.enc4(self.pool(e3)))
        b  = self.drop(self.bottleneck(self.pool(e4)))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ──────────────────────────────────────────────────────────
# 3D UNets
# MaxPool3d((1,2,2)) preserves depth=4 throughout.
# ──────────────────────────────────────────────────────────

class UNet3DSmall(nn.Module):
    """3D UNet. Filters: 16→32→64, bottleneck 128. Input [B,1,4,H,W]."""

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.enc1 = double_conv_3d(in_channels, 16)
        self.enc2 = double_conv_3d(16, 32)
        self.enc3 = double_conv_3d(32, 64)
        self.bottleneck = double_conv_3d(64, 128)
        self.up3  = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = double_conv_3d(128, 64)
        self.up2  = nn.ConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = double_conv_3d(64, 32)
        self.up1  = nn.ConvTranspose3d(32, 16, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = double_conv_3d(32, 16)
        self.out  = nn.Conv3d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)   # [B, NC, 4, H, W]


class UNet3DMedium(nn.Module):
    """3D UNet. Filters: 32→64→128→256, bottleneck 512. Input [B,1,4,H,W]."""

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.enc1 = double_conv_3d(in_channels, 32)
        self.enc2 = double_conv_3d(32, 64)
        self.enc3 = double_conv_3d(64, 128)
        self.enc4 = double_conv_3d(128, 256)
        self.bottleneck = double_conv_3d(256, 512)
        self.up4  = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
        self.dec4 = double_conv_3d(512, 256)
        self.up3  = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = double_conv_3d(256, 128)
        self.up2  = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = double_conv_3d(128, 64)
        self.up1  = nn.ConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = double_conv_3d(64, 32)
        self.out  = nn.Conv3d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)   # [B, NC, 4, H, W]


class UNet3DDropout(nn.Module):
    """3D UNet + Dropout3d. Filters: 32→64→128→256, bottleneck 512."""

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.drop = nn.Dropout3d(dropout)
        self.enc1 = double_conv_3d(in_channels, 32)
        self.enc2 = double_conv_3d(32, 64)
        self.enc3 = double_conv_3d(64, 128)
        self.enc4 = double_conv_3d(128, 256)
        self.bottleneck = double_conv_3d(256, 512)
        self.up4  = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
        self.dec4 = double_conv_3d(512, 256)
        self.up3  = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = double_conv_3d(256, 128)
        self.up2  = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = double_conv_3d(128, 64)
        self.up1  = nn.ConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = double_conv_3d(64, 32)
        self.out  = nn.Conv3d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.drop(self.enc3(self.pool(e2)))
        e4 = self.drop(self.enc4(self.pool(e3)))
        b  = self.drop(self.bottleneck(self.pool(e4)))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)   # [B, NC, 4, H, W]


# ──────────────────────────────────────────────────────────
# Factories
# ──────────────────────────────────────────────────────────

def get_2d_models(device):
    return {
        "2D-Small":   UNet2DSmall().to(device),
        "2D-Medium":  UNet2DMedium().to(device),
        "2D-Dropout": UNet2DDropout().to(device),
    }


def get_3d_models(device):
    return {
        "3D-Small":   UNet3DSmall().to(device),
        "3D-Medium":  UNet3DMedium().to(device),
        "3D-Dropout": UNet3DDropout().to(device),
    }


# ──────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d2      = torch.randn(2, 1, 128, 128).to(device)
    d3      = torch.randn(2, 1, 4, 128, 128).to(device)

    print(f"Device: {device}\n")
    print(f"{'Model':<15} {'Input':<20} {'Output':<22} {'Params':>10}")
    print("-" * 72)

    for name, model in get_2d_models(device).items():
        out    = model(d2)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<15} {str(tuple(d2.shape)):<20} {str(tuple(out.shape)):<22} {params:>10,}")

    for name, model in get_3d_models(device).items():
        out    = model(d3)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<15} {str(tuple(d3.shape)):<20} {str(tuple(out.shape)):<22} {params:>10,}")
