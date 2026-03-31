"""
DataGen.py
----------
Synthetic brain MRI volume dataset for tumour-grade segmentation.

Each sample is a 4-slice axial volume generated from a single coherent
3D anatomy. The tumour is modelled as a nested 3D ellipsoid; each slice
samples a different cross-section, giving volumetric coherence.

Classes (inspired by BraTS labelling):
  0 – Background          (outside skull)
  1 – Skull / meninges
  2 – Normal brain tissue (WM + GM combined)
  3 – Peritumoral edema   (Grade I-II periphery)
  4 – Non-enhancing core  (Grade III)
  5 – Necrotic/enhancing  (Grade IV / GBM)

Intensities approximate a T1-post-contrast sequence:
  skull bright, enhancing tumour very bright, necrosis dark.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE  = 128
N_SLICES    = 4          # consecutive axial slices per volume
NUM_CLASSES = 6
SLICE_PITCH = 7          # conceptual z-spacing in pixels between slices

CLASS_NAMES = [
    "Background",
    "Skull",
    "Normal Brain",
    "Peritumoral Edema",
    "Tumor Core (non-enh.)",
    "Necrotic/Enhancing",
]

# T1ce approximate pixel intensity for classes 1-5
GREY_SHADES = [200, 130, 70, 100, 240]


# ──────────────────────────────────────────────────────────
# Primitive helpers
# ──────────────────────────────────────────────────────────

def _ellipse(yy, xx, cy, cx, ry, rx):
    """Boolean mask for an axis-aligned ellipse."""
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _perturb(yy, xx, cy, cx, r, seeds):
    """
    Circle mask with sinusoidally perturbed radius — creates organic,
    tumour-like irregular boundaries.  `seeds` is a list of (phase, amp)
    tuples, one per frequency harmonic, shared across slices so the
    tumour shape is consistent through the volume.
    """
    angles = np.arctan2(yy.astype(float) - cy, xx.astype(float) - cx)
    dist   = np.sqrt((yy.astype(float) - cy) ** 2 + (xx.astype(float) - cx) ** 2)
    r_eff  = r * (1.0 + sum(amp * np.sin((k + 1) * angles + phi)
                             for k, (phi, amp) in enumerate(seeds)))
    r_eff  = np.maximum(r_eff, r * 0.15)   # prevent negative radii
    return dist <= r_eff


def _make_seeds(n=6, amp_max=0.15):
    return [(np.random.uniform(0, 2 * np.pi),
             np.random.uniform(0.02, amp_max / (k + 1)))
            for k in range(n)]


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

class BrainVolumeDataset(Dataset):
    """
    Returns (volume, mask) where:
      volume : float32 tensor [N_SLICES, H, W]  normalised to [0, 1]
      mask   : int64   tensor [N_SLICES, H, W]  class labels 0-5
    """

    def __init__(self, num_samples=2000, image_size=IMAGE_SIZE, tumor_prob=0.85):
        self.num_samples = num_samples
        self.image_size  = image_size
        self.tumor_prob  = tumor_prob

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        vol, msk = self._generate_volume()
        return (torch.from_numpy(vol).float() / 255.0,  # [S, H, W]
                torch.from_numpy(msk).long())            # [S, H, W]

    # ── volume generator ───────────────────────────────────

    def _generate_volume(self):
        H, W = self.image_size, self.image_size

        # Shared brain geometry
        cy = H / 2 + np.random.uniform(-H * 0.04, H * 0.04)
        cx = W / 2 + np.random.uniform(-W * 0.04, W * 0.04)
        ry = np.random.uniform(H * 0.36, H * 0.43)
        rx = np.random.uniform(ry * 0.85, ry * 1.05)
        skull_thick = np.random.uniform(0.06, 0.09)

        # Tumour parameters (coherent across slices)
        tumor = None
        if np.random.random() < self.tumor_prob:
            tc_z  = np.random.uniform(0.5, 2.5)

            # Place tumour centre inside the brain parenchyma
            safe_ry = ry * (1 - skull_thick) * 0.60
            safe_rx = rx * (1 - skull_thick) * 0.60
            angle   = np.random.uniform(0, 2 * np.pi)
            dist    = np.random.uniform(0, 0.85)
            tc_y    = cy + dist * safe_ry * np.sin(angle)
            tc_x    = cx + dist * safe_rx * np.cos(angle)

            r_edema = np.random.uniform(14, 28)
            r_tumor = r_edema * np.random.uniform(0.40, 0.62)
            r_nec   = r_tumor * np.random.uniform(0.30, 0.55)

            tumor = dict(
                tc_z=tc_z, tc_y=tc_y, tc_x=tc_x,
                r_edema=r_edema, r_tumor=r_tumor, r_nec=r_nec,
                seeds_edema=_make_seeds(amp_max=0.18),
                seeds_tumor=_make_seeds(amp_max=0.14),
                seeds_nec  =_make_seeds(amp_max=0.10),
            )

        slices, masks = [], []
        for z in range(N_SLICES):
            img, msk = self._generate_slice(z, cy, cx, ry, rx, skull_thick, tumor)
            slices.append(img)
            masks.append(msk)

        return np.stack(slices), np.stack(masks)   # [S,H,W] each

    # ── single slice ───────────────────────────────────────

    def _generate_slice(self, z, cy, cx, ry, rx, skull_thick, tumor):
        H, W  = self.image_size, self.image_size
        image = np.zeros((H, W), dtype=np.float32)
        mask  = np.zeros((H, W), dtype=np.int64)
        yy, xx = np.mgrid[0:H, 0:W]

        # Tiny per-slice motion jitter (simulate patient breathing)
        sc_y = cy + np.random.uniform(-1.5, 1.5)
        sc_x = cx + np.random.uniform(-1.5, 1.5)

        # Class 1 – skull ring
        skull = _ellipse(yy, xx, sc_y, sc_x, ry, rx)
        image[skull] = GREY_SHADES[0]
        mask[skull]  = 1

        # Class 2 – brain tissue (inner ellipse)
        ri_y  = ry * (1 - skull_thick)
        ri_x  = rx * (1 - skull_thick)
        brain = _ellipse(yy, xx, sc_y, sc_x, ri_y, ri_x)
        image[brain] = GREY_SHADES[1]
        mask[brain]  = 2

        # Tumour layers (outermost → innermost, painter's model)
        if tumor is not None:
            dz   = (z - tumor['tc_z']) * SLICE_PITCH   # z-offset in pixels
            tc_y = tumor['tc_y']
            tc_x = tumor['tc_x']

            for r_key, cls, shade_idx, seeds_key, clip_to_brain in [
                ('r_edema', 3, 2, 'seeds_edema', True),
                ('r_tumor', 4, 3, 'seeds_tumor', False),
                ('r_nec',   5, 4, 'seeds_nec',   False),
            ]:
                r_3d = tumor[r_key]
                r_z  = float(np.sqrt(max(0.0, r_3d ** 2 - dz ** 2)))
                if r_z < 1.5:
                    continue
                blob = _perturb(yy, xx, tc_y, tc_x, r_z, tumor[seeds_key])
                if clip_to_brain:
                    blob &= brain
                image[blob] = GREY_SHADES[shade_idx]
                mask[blob]  = cls

        # Per-slice acquisition noise + mild intensity bias
        noise = np.random.normal(0, 5, (H, W))
        bias  = np.random.uniform(-8, 8)
        image = np.clip(image + noise + bias, 0, 255).astype(np.uint8)
        return image, mask


# ──────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────

def get_dataloaders(train_size=2000, val_size=400, test_size=400,
                    batch_size=12, image_size=IMAGE_SIZE):
    """Return (train, val, test) DataLoaders. Batch shape: [B, 4, H, W]."""
    kw = dict(num_workers=0, pin_memory=False)
    return (
        DataLoader(BrainVolumeDataset(train_size, image_size),
                   batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(BrainVolumeDataset(val_size,   image_size),
                   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(BrainVolumeDataset(test_size,  image_size),
                   batch_size=batch_size, shuffle=False, **kw),
    )


# ──────────────────────────────────────────────────────────
# Preview
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    N    = 3    # number of volumes to preview
    CMAP = plt.colormaps["tab10"].resampled(NUM_CLASSES)
    ds   = BrainVolumeDataset(num_samples=N, tumor_prob=0.95)

    # Layout: for each volume, one row of images and one row of masks
    fig, axes = plt.subplots(N * 2, N_SLICES, figsize=(3 * N_SLICES, 3.5 * N * 2))

    for i in range(N):
        vol, msk = ds[i]
        ri = i * 2      # image row
        rm = i * 2 + 1  # mask row
        for z in range(N_SLICES):
            axes[ri, z].imshow(vol[z].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[ri, z].set_title(f"Vol {i}  Slice {z}", fontsize=8)
            axes[ri, z].axis("off")

            axes[rm, z].imshow(msk[z].numpy(), cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
            axes[rm, z].axis("off")

    patches = [mpatches.Patch(color=CMAP(c), label=f"{c}: {CLASS_NAMES[c]}")
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("BrainVolumeDataset  (odd rows = image, even rows = segmentation mask)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("datagen_preview.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Saved datagen_preview.png")

    # DataLoader smoke test
    tl, vl, te = get_dataloaders(train_size=64, val_size=16, test_size=16, batch_size=4)
    v, m = next(iter(tl))
    print(f"\nBatch — volume: {tuple(v.shape)} {v.dtype}  "
          f"mask: {tuple(m.shape)} {m.dtype}  "
          f"classes: {sorted(m.unique().tolist())}")
