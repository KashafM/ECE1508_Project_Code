import torch
import torch.nn as nn

from DataGen import NUM_CLASSES


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

def double_conv(in_ch, out_ch):
    """Two 3×3 conv layers with BN and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Model 1 – UNetSmall
# Filters: 16 → 32 → 64 | bottleneck 128 | 3 encoder levels
# Lightest model, fastest to train.
# ---------------------------------------------------------------------------

class UNetSmall(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = double_conv(in_channels, 16)
        self.enc2 = double_conv(16, 32)
        self.enc3 = double_conv(32, 64)

        # Bottleneck
        self.bottleneck = double_conv(64, 128)

        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = double_conv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = double_conv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = double_conv(32, 16)

        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Model 2 – UNetMedium
# Filters: 32 → 64 → 128 → 256 | bottleneck 512 | 4 encoder levels
# Standard UNet capacity.
# ---------------------------------------------------------------------------

class UNetMedium(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = double_conv(in_channels, 32)
        self.enc2 = double_conv(32, 64)
        self.enc3 = double_conv(64, 128)
        self.enc4 = double_conv(128, 256)

        # Bottleneck
        self.bottleneck = double_conv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Model 3 – UNetDropout
# Same depth as UNetMedium but with Dropout2d in the encoder/bottleneck.
# Regularised variant to compare generalisation against UNetMedium.
# ---------------------------------------------------------------------------

class UNetDropout(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(p=dropout)

        # Encoder
        self.enc1 = double_conv(in_channels, 32)
        self.enc2 = double_conv(32, 64)
        self.enc3 = double_conv(64, 128)
        self.enc4 = double_conv(128, 256)

        # Bottleneck
        self.bottleneck = double_conv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.drop(self.enc3(self.pool(e2)))
        e4 = self.drop(self.enc4(self.pool(e3)))

        b = self.drop(self.bottleneck(self.pool(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_models(device):
    """Return a dict of all three models, moved to device."""
    return {
        "UNet-Small":   UNetSmall().to(device),
        "UNet-Medium":  UNetMedium().to(device),
        "UNet-Dropout": UNetDropout().to(device),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 1, 128, 128).to(device)
    for name, model in get_models(device).items():
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:15s}  output: {tuple(out.shape)}  params: {params:,}")
