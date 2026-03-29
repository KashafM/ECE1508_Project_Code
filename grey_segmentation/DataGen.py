import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE = 128
NUM_CLASSES = 6  # 0 = background (black), 1-5 = five tissue shades
GREY_SHADES = [40, 80, 120, 170, 220]  # pixel intensities per class 1-5

# Anatomical meaning of each class (T1-weighted MRI approximation):
#   0 – background / air        (black,   ~0)
#   1 – skull / meninges        (darkest, ~40)
#   2 – cerebrospinal fluid     (dark,    ~80)
#   3 – grey matter             (mid,     ~120)
#   4 – white matter            (light,   ~170)
#   5 – deep nuclei / lesions   (bright,  ~220)
CLASS_NAMES = [
    "Background",
    "Skull/Meninges",
    "CSF",
    "Grey Matter",
    "White Matter",
    "Deep Nuclei",
]


def _ellipse_mask(yy, xx, cy, cx, ry, rx):
    """Boolean mask for an axis-aligned ellipse."""
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _add_noise(image, sigma=4):
    """Add mild Gaussian noise to simulate MRI acquisition noise."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


class GreyCircleDataset(Dataset):
    """
    Synthesises simple T1-weighted brain MRI slices for segmentation.

    Layout (painter's model – later regions overwrite earlier ones):
      Class 0 – black background (air outside the head)
      Class 1 – skull ellipse:  large outer ellipse, thin ring
      Class 2 – CSF:            slightly smaller ellipse just inside skull
      Class 3 – grey matter:    irregular blobs around the cortex band
      Class 4 – white matter:   inner ellipse filling the central brain
      Class 5 – deep nuclei:    one or two small bright blobs near centre

    Random variation per sample:
      • overall brain size / aspect ratio
      • slight centre jitter
      • number and placement of grey-matter blobs and deep-nuclei blobs
      • mild Gaussian noise on the pixel values
    """

    def __init__(self, num_samples=2000, image_size=IMAGE_SIZE):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask = self._generate()
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

    def _generate(self):
        H, W = self.image_size, self.image_size
        image = np.zeros((H, W), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.int64)   # class 0 = background

        yy, xx = np.mgrid[0:H, 0:W]

        # Brain centre with small random jitter
        cy = H / 2 + np.random.uniform(-H * 0.05, H * 0.05)
        cx = W / 2 + np.random.uniform(-W * 0.05, W * 0.05)

        # Outer skull radii (ellipse, slightly wider than tall like a real head)
        ry_skull = np.random.uniform(H * 0.36, H * 0.44)
        rx_skull = np.random.uniform(ry_skull * 0.85, ry_skull * 1.05)

        skull_thickness  = np.random.uniform(0.06, 0.10)   # fraction of ry_skull
        csf_thickness    = np.random.uniform(0.06, 0.10)
        gm_thickness     = np.random.uniform(0.12, 0.18)

        ry_csf = ry_skull * (1 - skull_thickness)
        rx_csf = rx_skull * (1 - skull_thickness)

        ry_gm  = ry_csf  * (1 - csf_thickness)
        rx_gm  = rx_csf  * (1 - csf_thickness)

        ry_wm  = ry_gm   * (1 - gm_thickness)
        rx_wm  = rx_gm   * (1 - gm_thickness)

        # --- draw layers from outermost to innermost ---

        # Class 1 – skull
        skull_mask = _ellipse_mask(yy, xx, cy, cx, ry_skull, rx_skull)
        image[skull_mask] = GREY_SHADES[0]
        mask[skull_mask]  = 1

        # Class 2 – CSF
        csf_mask = _ellipse_mask(yy, xx, cy, cx, ry_csf, rx_csf)
        image[csf_mask] = GREY_SHADES[1]
        mask[csf_mask]  = 2

        # Class 3 – grey matter: blobs placed on the cortical band
        n_gm_blobs = np.random.randint(6, 14)
        for _ in range(n_gm_blobs):
            angle = np.random.uniform(0, 2 * np.pi)
            # Blob centre sits within the CSF–WM band
            dist_frac = np.random.uniform(0.55, 0.90)
            bcy = cy + dist_frac * ry_gm * np.sin(angle)
            bcx = cx + dist_frac * rx_gm * np.cos(angle)
            br  = np.random.uniform(ry_skull * 0.08, ry_skull * 0.18)
            blob = (yy - bcy) ** 2 + (xx - bcx) ** 2 <= br ** 2
            image[blob] = GREY_SHADES[2]
            mask[blob]  = 3

        # Class 4 – white matter: solid inner ellipse
        wm_mask = _ellipse_mask(yy, xx, cy, cx, ry_wm, rx_wm)
        image[wm_mask] = GREY_SHADES[3]
        mask[wm_mask]  = 4

        # Class 5 – deep nuclei: 1-3 small bright blobs near centre
        n_nuclei = np.random.randint(1, 4)
        for _ in range(n_nuclei):
            angle = np.random.uniform(0, 2 * np.pi)
            dist_frac = np.random.uniform(0.0, 0.40)
            ncy = cy + dist_frac * ry_wm * np.sin(angle)
            ncx = cx + dist_frac * rx_wm * np.cos(angle)
            nr  = np.random.uniform(ry_skull * 0.05, ry_skull * 0.10)
            nucleus = (yy - ncy) ** 2 + (xx - ncx) ** 2 <= nr ** 2
            image[nucleus] = GREY_SHADES[4]
            mask[nucleus]  = 5

        # Mild acquisition noise (applied to pixel values only, not mask)
        image = _add_noise(image, sigma=4)

        return image, mask


def get_dataloaders(train_size=2000, val_size=400, test_size=400,
                    batch_size=16, image_size=IMAGE_SIZE):
    """Return train, val, and test DataLoaders."""
    train_ds = GreyCircleDataset(train_size, image_size)
    val_ds   = GreyCircleDataset(val_size,   image_size)
    test_ds  = GreyCircleDataset(test_size,  image_size)

    kwargs = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    N = 6  # samples to display
    CMAP = plt.colormaps["tab10"].resampled(NUM_CLASSES)

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
