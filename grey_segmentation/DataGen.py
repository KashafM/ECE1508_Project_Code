import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE = 128
NUM_CLASSES = 6  # 0 = background (black), 1-5 = five grey shades
GREY_SHADES = [51, 102, 153, 204, 255]  # pixel intensities for classes 1-5


class GreyCircleDataset(Dataset):
    """
    Generates images of overlapping grey circles on a black background.
    Each pixel is assigned one of 6 classes:
      0 = background (black)
      1 = darkest grey  (~51)
      2 = dark grey     (~102)
      3 = mid grey      (~153)
      4 = light grey    (~204)
      5 = lightest grey (~255)
    Later-drawn circles occlude earlier ones (painter's model).
    """

    def __init__(self, num_samples=2000, image_size=IMAGE_SIZE,
                 num_circles_range=(6, 20)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_circles_range = num_circles_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask = self._generate()
        # image: [1, H, W] float in [0, 1]
        # mask:  [H, W]    long  with class indices 0-5
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

    def _generate(self):
        H, W = self.image_size, self.image_size
        image = np.zeros((H, W), dtype=np.uint8)   # black background
        mask = np.zeros((H, W), dtype=np.int64)    # class 0 = background

        yy, xx = np.mgrid[0:H, 0:W]
        num_circles = np.random.randint(*self.num_circles_range)

        for _ in range(num_circles):
            cls = np.random.randint(1, NUM_CLASSES)   # 1-5
            shade = GREY_SHADES[cls - 1]
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)
            r = np.random.randint(8, H // 3)
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            image[inside] = shade
            mask[inside] = cls

        return image, mask


def get_dataloaders(train_size=2000, val_size=400, test_size=400,
                    batch_size=16, image_size=IMAGE_SIZE):
    """Return train, val, and test DataLoaders."""
    train_ds = GreyCircleDataset(train_size, image_size)
    val_ds = GreyCircleDataset(val_size, image_size)
    test_ds = GreyCircleDataset(test_size, image_size)

    kwargs = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    N = 6  # samples to display
    CLASS_NAMES = ["Background"] + [f"Grey-{v}" for v in GREY_SHADES]
    CMAP = plt.cm.get_cmap("tab10", NUM_CLASSES)

    ds = GreyCircleDataset(num_samples=N)
    print(f"Dataset length : {len(ds)}")
    print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

    # --- sample images + masks ---
    fig, axes = plt.subplots(3, N, figsize=(3 * N, 9))
    for i in range(N):
        img, msk = ds[i]

        axes[0, i].imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Sample {i}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(msk.numpy(), cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
        axes[1, i].set_title("Mask", fontsize=9)
        axes[1, i].axis("off")

        # class distribution bar
        counts = [(msk == c).sum().item() for c in range(NUM_CLASSES)]
        total = sum(counts)
        fracs = [c / total for c in counts]
        axes[2, i].bar(range(NUM_CLASSES), fracs,
                       color=[CMAP(c) for c in range(NUM_CLASSES)],
                       edgecolor="black", linewidth=0.4)
        axes[2, i].set_xticks(range(NUM_CLASSES))
        axes[2, i].set_xticklabels([str(c) for c in range(NUM_CLASSES)], fontsize=7)
        axes[2, i].set_ylabel("Pixel fraction", fontsize=7)
        axes[2, i].set_title("Class dist.", fontsize=9)

    # legend
    patches = [mpatches.Patch(color=CMAP(c), label=f"{c}: {CLASS_NAMES[c]}")
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("DataGen preview  (row 0: image | row 1: mask | row 2: class distribution)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("datagen_preview.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved datagen_preview.png")

    # --- quick DataLoader smoke test ---
    train_loader, val_loader, test_loader = get_dataloaders(
        train_size=100, val_size=20, test_size=20, batch_size=8
    )
    batch_img, batch_msk = next(iter(train_loader))
    print(f"\nDataLoader batch shapes:")
    print(f"  images : {tuple(batch_img.shape)}  dtype={batch_img.dtype}")
    print(f"  masks  : {tuple(batch_msk.shape)}  dtype={batch_msk.dtype}")
    print(f"  unique classes in batch: {sorted(batch_msk.unique().tolist())}")
