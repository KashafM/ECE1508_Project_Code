"""
DataGen.py
----------
Synthetic T1-post-contrast brain MRI volumes for tumour-grade segmentation.

Realism features
----------------
* Gyral surface  – brain boundary is a sinusoidally-perturbed ellipse; the
                   gap between the smooth skull inner wall and the bumpy brain
                   surface is filled with dark CSF, naturally producing sulci.
* WM / GM layers – two-tone brain: bright inner white matter, darker grey-
                   matter cortex that follows the irregular gyral boundary.
* Substructures  – lateral ventricles (L-shaped: body + temporal horn),
                   thalami, basal ganglia, corpus callosum, brainstem.
* MRI effects    – spatially-smooth WM texture, cosine bias field
                   (intensity inhomogeneity), Gaussian blur (partial-volume),
                   per-slice Gaussian noise + intensity offset.
* Tumour         – nested 3D ellipsoids (edema / core / necrosis) with
                   irregular sinusoidal boundary; cross-section evolves
                   coherently across the 4 axial slices.

Classes
-------
  0 – Background
  1 – Skull / meninges
  2 – Normal brain tissue  (CSF layer + GM + WM + substructures)
  3 – Peritumoral edema    (Grade I–II)
  4 – Tumour core          (Grade III, non-enhancing)
  5 – Necrotic/enhancing   (Grade IV / GBM)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

IMAGE_SIZE  = 128
N_SLICES    = 4
NUM_CLASSES = 6
SLICE_PITCH = 7   # effective z-spacing in pixels

CLASS_NAMES = [
    "Background",
    "Skull",
    "Normal Brain",
    "Peritumoral Edema",
    "Tumor Core (non-enh.)",
    "Necrotic/Enhancing",
]


# ──────────────────────────────────────────────────────────
# Primitive helpers
# ──────────────────────────────────────────────────────────

def _ellipse(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _gyri_mask(yy, xx, cy, cx, ry, rx, freqs, amps, phases):
    """
    Perturbed ellipse that mimics the brain's gyral surface.
    Each frequency adds a sinusoidal ripple to the ellipse boundary:
      low  freqs (3–7)  → major gyral folds  (~2–4 px amplitude)
      high freqs (8–18) → fine sulcal texture (~0.5–1.5 px)
    The result is a realistic bumpy brain outline.
    """
    dy = yy.astype(float) - cy
    dx = xx.astype(float) - cx
    angles   = np.arctan2(dy, dx)
    d_norm   = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    r_bound  = np.ones_like(angles)
    for f, a, p in zip(freqs, amps, phases):
        r_bound += a * np.sin(f * angles + p)
    return d_norm <= r_bound


def _perturb(yy, xx, cy, cx, r, seeds):
    """Tumour blob: circle with sinusoidal radius perturbation."""
    angles = np.arctan2(yy.astype(float) - cy, xx.astype(float) - cx)
    dist   = np.sqrt((yy.astype(float) - cy) ** 2 + (xx.astype(float) - cx) ** 2)
    r_eff  = r * (1.0 + sum(amp * np.sin((k + 1) * angles + phi)
                             for k, (phi, amp) in enumerate(seeds)))
    r_eff  = np.maximum(r_eff, r * 0.15)
    return dist <= r_eff


def _make_seeds(n=6, amp_max=0.15):
    return [(np.random.uniform(0, 2 * np.pi),
             np.random.uniform(0.02, amp_max / (k + 1)))
            for k in range(n)]


def _bias_field(H, W):
    """
    Smooth multiplicative MRI intensity inhomogeneity (bias field).
    Uses a sum of low-frequency cosines — mimics B1 field non-uniformity.
    """
    x  = np.linspace(0, 2 * np.pi, W)
    y  = np.linspace(0, 2 * np.pi, H)
    XX, YY = np.meshgrid(x, y)
    field = (1.0
             + 0.05 * np.random.randn() * np.cos(0.5 * XX + np.random.uniform(0, np.pi))
             + 0.05 * np.random.randn() * np.cos(0.5 * YY + np.random.uniform(0, np.pi))
             + 0.03 * np.random.randn() * np.cos(XX + YY   + np.random.uniform(0, np.pi)))
    return field


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

class BrainVolumeDataset(Dataset):
    """
    Each sample:  (volume [4, H, W] float32 / 255,  mask [4, H, W] int64)
    Tumour prevalence defaults to 85 %.
    """

    def __init__(self, num_samples=2000, image_size=IMAGE_SIZE, tumor_prob=0.85):
        self.num_samples = num_samples
        self.image_size  = image_size
        self.tumor_prob  = tumor_prob

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        vol, msk = self._generate_volume()
        return torch.from_numpy(vol).float() / 255.0, torch.from_numpy(msk).long()

    # ── volume ─────────────────────────────────────────────

    def _generate_volume(self):
        H, W = self.image_size, self.image_size

        # Overall brain size / position
        cy = H / 2 + np.random.uniform(-H * 0.04, H * 0.04)
        cx = W / 2 + np.random.uniform(-W * 0.04, W * 0.04)
        ry = np.random.uniform(H * 0.36, H * 0.43)
        rx = np.random.uniform(ry * 0.85, ry * 1.05)
        skull_thick = np.random.uniform(0.06, 0.09)

        # Gyral surface: blend of low- and high-frequency sinusoids
        # (shared across slices so anatomy is self-consistent)
        low_f  = list(range(3, 8))                       # major folds
        high_f = list(np.random.choice(range(8, 19), 10, replace=False))
        freqs  = low_f + high_f
        amps   = ([np.random.uniform(0.030, 0.055) for _ in low_f] +
                  [np.random.uniform(0.010, 0.022) for _ in high_f])
        phases = np.random.uniform(0, 2 * np.pi, len(freqs)).tolist()

        # Inner brain substructure geometry (fractions of ri)
        bp = dict(
            gyri_freqs=freqs, gyri_amps=amps, gyri_phases=phases,

            # Lateral ventricle: body ellipse + temporal horn per side
            vb_ry=np.random.uniform(0.09, 0.15), vb_rx=np.random.uniform(0.06, 0.11),
            vb_dy=np.random.uniform(-0.18, 0.00), vb_dx=np.random.uniform(0.10, 0.20),
            vh_ry=np.random.uniform(0.06, 0.10), vh_rx=np.random.uniform(0.04, 0.08),
            vh_dy=np.random.uniform(0.05, 0.15), vh_dx=np.random.uniform(0.08, 0.18),

            # Thalami
            th_ry=np.random.uniform(0.06, 0.10), th_rx=np.random.uniform(0.06, 0.10),
            th_dy=np.random.uniform(-0.04, 0.08), th_dx=np.random.uniform(0.06, 0.13),

            # Basal ganglia
            bg_ry=np.random.uniform(0.05, 0.09), bg_rx=np.random.uniform(0.06, 0.10),
            bg_dy=np.random.uniform(0.01, 0.10), bg_dx=np.random.uniform(0.16, 0.25),

            # Corpus callosum
            cc_ry=np.random.uniform(0.03, 0.055), cc_rx=np.random.uniform(0.24, 0.38),
            cc_dy=np.random.uniform(-0.30, -0.10),

            # Brainstem
            bs_ry=np.random.uniform(0.15, 0.22), bs_rx=np.random.uniform(0.08, 0.13),
            bs_dy=np.random.uniform(0.48, 0.64), bs_dx=np.random.uniform(-0.04, 0.04),
        )

        # Tumour
        tumor = None
        if np.random.random() < self.tumor_prob:
            tc_z    = np.random.uniform(0.5, 2.5)
            safe_ry = ry * (1 - skull_thick) * 0.58
            safe_rx = rx * (1 - skull_thick) * 0.58
            angle   = np.random.uniform(0, 2 * np.pi)
            dist    = np.random.uniform(0, 0.85)
            tc_y    = cy + dist * safe_ry * np.sin(angle)
            tc_x    = cx + dist * safe_rx * np.cos(angle)
            r_edema = np.random.uniform(13, 27)
            r_tumor = r_edema * np.random.uniform(0.40, 0.62)
            r_nec   = r_tumor  * np.random.uniform(0.30, 0.55)
            tumor   = dict(
                tc_z=tc_z, tc_y=tc_y, tc_x=tc_x,
                r_edema=r_edema, r_tumor=r_tumor, r_nec=r_nec,
                seeds_edema=_make_seeds(amp_max=0.18),
                seeds_tumor=_make_seeds(amp_max=0.14),
                seeds_nec  =_make_seeds(amp_max=0.10),
            )

        slices, masks = [], []
        for z in range(N_SLICES):
            img, msk = self._generate_slice(z, cy, cx, ry, rx, skull_thick, bp, tumor)
            slices.append(img)
            masks.append(msk)
        return np.stack(slices), np.stack(masks)

    # ── single slice ───────────────────────────────────────

    def _generate_slice(self, z, cy, cx, ry, rx, skull_thick, bp, tumor):
        H, W   = self.image_size, self.image_size
        image  = np.zeros((H, W), dtype=np.float32)
        mask   = np.zeros((H, W), dtype=np.int64)
        yy, xx = np.mgrid[0:H, 0:W]

        # Tiny per-slice motion jitter
        sc_y = cy + np.random.uniform(-1.5, 1.5)
        sc_x = cx + np.random.uniform(-1.5, 1.5)

        # ── Class 1: skull bone ring ───────────────────────────────────────
        ri_y = ry * (1 - skull_thick)
        ri_x = rx * (1 - skull_thick)
        skull_outer = _ellipse(yy, xx, sc_y, sc_x, ry, rx)
        skull_inner = _ellipse(yy, xx, sc_y, sc_x, ri_y, ri_x)
        skull_ring  = skull_outer & ~skull_inner
        image[skull_ring] = 195 + np.random.uniform(-8, 8)
        mask[skull_ring]  = 1

        # ── Gyral brain surface ────────────────────────────────────────────
        # Brain boundary sits just inside the skull inner wall.
        # The gap between skull_inner and this perturbed surface = subarachnoid
        # CSF + sulci — naturally creates realistic cortical folding.
        gyri_scale = 0.955   # brain fills ~95.5% of skull inner radius
        brain = _gyri_mask(yy, xx, sc_y, sc_x,
                           ri_y * gyri_scale, ri_x * gyri_scale,
                           bp['gyri_freqs'], bp['gyri_amps'], bp['gyri_phases'])
        brain &= skull_inner

        # Subarachnoid CSF / sulci  (dark, between skull inner & brain)
        csf_sulci = skull_inner & ~brain
        image[csf_sulci] = np.random.uniform(28, 46)
        mask[csf_sulci]  = 2

        # ── Grey matter cortex (outer ~15 % of brain depth) ───────────────
        # WM lives inside a smooth ellipse at ~83 % of brain radius.
        wm_ry = ri_y * gyri_scale * 0.83
        wm_rx = ri_x * gyri_scale * 0.83
        wm_ellipse = _ellipse(yy, xx, sc_y, sc_x, wm_ry, wm_rx)

        grey_matter = brain & ~wm_ellipse
        # GM base intensity + subtle per-pixel variation
        gm_base  = np.random.uniform(88, 100)
        gm_noise = gaussian_filter(np.random.normal(0, 5, (H, W)), sigma=3)
        image[grey_matter] = np.clip(gm_base + gm_noise[grey_matter], 70, 118)
        mask[grey_matter]  = 2

        # ── White matter interior ──────────────────────────────────────────
        wm_base  = np.random.uniform(140, 158)
        wm_noise = gaussian_filter(np.random.normal(0, 8, (H, W)), sigma=10)
        image[wm_ellipse] = np.clip(wm_base + wm_noise[wm_ellipse], 115, 175)
        mask[wm_ellipse]  = 2

        # ── Lateral ventricles: body + temporal horn per side ─────────────
        vb_cy  = sc_y + ri_y * bp['vb_dy']
        vh_cy  = sc_y + ri_y * bp['vh_dy']
        vb_ry_ = ri_y * bp['vb_ry'];  vb_rx_ = ri_x * bp['vb_rx']
        vh_ry_ = ri_y * bp['vh_ry'];  vh_rx_ = ri_x * bp['vh_rx']

        vents = np.zeros((H, W), dtype=bool)
        for side in (-1, 1):
            vb_cx = sc_x + side * ri_x * bp['vb_dx']
            vh_cx = sc_x + side * ri_x * bp['vh_dx']
            v = (_ellipse(yy, xx, vb_cy, vb_cx, vb_ry_, vb_rx_) |
                 _ellipse(yy, xx, vh_cy, vh_cx, vh_ry_, vh_rx_)) & wm_ellipse
            image[v] = np.random.uniform(25, 42)
            vents   |= v
        mask[vents] = 2

        # ── Thalami ────────────────────────────────────────────────────────
        th_cy = sc_y + ri_y * bp['th_dy']
        for side in (-1, 1):
            th_cx = sc_x + side * ri_x * bp['th_dx']
            th = _ellipse(yy, xx, th_cy, th_cx,
                          ri_y * bp['th_ry'], ri_x * bp['th_rx']) & wm_ellipse & ~vents
            image[th] = np.random.uniform(112, 126)

        # ── Basal ganglia ──────────────────────────────────────────────────
        bg_cy = sc_y + ri_y * bp['bg_dy']
        for side in (-1, 1):
            bg_cx = sc_x + side * ri_x * bp['bg_dx']
            bg = _ellipse(yy, xx, bg_cy, bg_cx,
                          ri_y * bp['bg_ry'], ri_x * bp['bg_rx']) & wm_ellipse & ~vents
            image[bg] = np.random.uniform(100, 116)

        # ── Corpus callosum ────────────────────────────────────────────────
        cc_cy = sc_y + ri_y * bp['cc_dy']
        cc    = _ellipse(yy, xx, cc_cy, sc_x,
                         ri_y * bp['cc_ry'], ri_x * bp['cc_rx']) & wm_ellipse & ~vents
        image[cc] = np.random.uniform(155, 170)

        # ── Brainstem ──────────────────────────────────────────────────────
        bs_cy = sc_y + ri_y * bp['bs_dy']
        bs_cx = sc_x + ri_x * bp['bs_dx']
        bs    = _ellipse(yy, xx, bs_cy, bs_cx,
                         ri_y * bp['bs_ry'], ri_x * bp['bs_rx']) & skull_inner
        image[bs] = np.random.uniform(110, 126)
        mask[bs & ~brain]  = 2   # brainstem extends outside brain → still class 2
        mask[bs &  brain]  = 2   # already 2

        # ── Bias field (smooth intensity inhomogeneity) ────────────────────
        image *= _bias_field(H, W)

        # ── Gaussian blur (partial-volume effect at boundaries) ────────────
        image = gaussian_filter(image, sigma=0.9)

        # ── Tumour layers (painted on top after blur) ──────────────────────
        if tumor is not None:
            dz   = (z - tumor['tc_z']) * SLICE_PITCH
            tc_y = tumor['tc_y']
            tc_x = tumor['tc_x']
            for r_key, cls, intensity, seeds_key, clip in [
                ('r_edema', 3,  68, 'seeds_edema', True),
                ('r_tumor', 4,  98, 'seeds_tumor', False),
                ('r_nec',   5, 238, 'seeds_nec',   False),
            ]:
                r_3d = tumor[r_key]
                r_z  = float(np.sqrt(max(0.0, r_3d ** 2 - dz ** 2)))
                if r_z < 1.5:
                    continue
                blob = _perturb(yy, xx, tc_y, tc_x, r_z, tumor[seeds_key])
                if clip:
                    blob &= brain
                image[blob] = intensity
                mask[blob]  = cls

        # ── Final per-slice noise ──────────────────────────────────────────
        image = np.clip(image + np.random.normal(0, 4, (H, W))
                        + np.random.uniform(-5, 5), 0, 255).astype(np.uint8)
        return image, mask


# ──────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────

def get_dataloaders(train_size=2000, val_size=400, test_size=400,
                    batch_size=12, image_size=IMAGE_SIZE):
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

    N    = 3
    CMAP = plt.colormaps["tab10"].resampled(NUM_CLASSES)
    ds   = BrainVolumeDataset(num_samples=N, tumor_prob=0.95)

    fig, axes = plt.subplots(N * 2, N_SLICES, figsize=(3 * N_SLICES, 3.5 * N * 2))
    for i in range(N):
        vol, msk = ds[i]
        for z in range(N_SLICES):
            axes[i*2,   z].imshow(vol[z].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i*2,   z].set_title(f"Vol {i}  z={z}", fontsize=8)
            axes[i*2,   z].axis("off")
            axes[i*2+1, z].imshow(msk[z].numpy(), cmap=CMAP, vmin=0, vmax=NUM_CLASSES-1)
            axes[i*2+1, z].axis("off")

    patches = [mpatches.Patch(color=CMAP(c), label=f"{c}: {CLASS_NAMES[c]}")
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("BrainVolumeDataset — synthetic T1ce MRI\n"
                 "(odd rows: image  |  even rows: segmentation mask)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("datagen_preview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved datagen_preview.png")
