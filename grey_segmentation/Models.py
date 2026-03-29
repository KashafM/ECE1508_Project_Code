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
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from DataGen import GreyCircleDataset, NUM_CLASSES, CLASS_NAMES

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QUICK_EPOCHS = 5
    QUICK_TRAIN  = 300
    BATCH        = 16
    N_SHOW       = 4          # images to visualise per model

    CMAP = plt.colormaps["tab10"].resampled(NUM_CLASSES)

    # ---- param count summary ------------------------------------------------
    print(f"Device: {DEVICE}\n")
    dummy = torch.randn(2, 1, 128, 128).to(DEVICE)
    for name, m in get_models(DEVICE).items():
        out = m(dummy)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:15s}  output: {tuple(out.shape)}  params: {params:,}")

    # ---- quick mini-train so predictions are meaningful --------------------
    from torch.utils.data import DataLoader

    train_ds = GreyCircleDataset(num_samples=QUICK_TRAIN)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    models = get_models(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nQuick-training each model for {QUICK_EPOCHS} epochs "
          f"on {QUICK_TRAIN} samples …")
    for name, model in models.items():
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(QUICK_EPOCHS):
            epoch_loss = 0.0
            for imgs, msks in train_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                opt.zero_grad()
                loss = criterion(model(imgs), msks)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            print(f"  {name}  epoch {epoch+1}/{QUICK_EPOCHS}  "
                  f"loss={epoch_loss/len(train_loader):.4f}")

    # ---- visualise predictions on fresh samples ----------------------------
    vis_ds = GreyCircleDataset(num_samples=N_SHOW)
    n_models = len(models)

    # Columns: input | ground truth | model-1 | model-2 | model-3
    n_cols = 2 + n_models
    fig, axes = plt.subplots(N_SHOW, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * N_SHOW))
    col_titles = ["Input", "Ground Truth"] + list(models.keys())

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    for row in range(N_SHOW):
        img, msk = vis_ds[row]

        # Input (greyscale)
        axes[row, 0].imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")

        # Ground truth mask
        axes[row, 1].imshow(msk.numpy(), cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 1].axis("off")

        # Each model's prediction
        for col, (name, model) in enumerate(models.items(), start=2):
            model.eval()
            with torch.no_grad():
                logits = model(img.unsqueeze(0).to(DEVICE))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            axes[row, col].imshow(pred, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
            axes[row, col].axis("off")

    # Shared legend
    patches = [mpatches.Patch(color=CMAP(c), label=f"{c}: {CLASS_NAMES[c]}")
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"Model predictions after {QUICK_EPOCHS}-epoch quick-train  "
                 f"(col 0: input | col 1: GT | rest: model outputs)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_preview.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\nSaved model_preview.png")
