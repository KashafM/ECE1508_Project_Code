"""
Analysis.py
-----------
Train and compare 2D vs 3D UNet models for brain tumour segmentation.

2D approach: each volume [B,4,H,W] is split into B*4 independent slices
             processed one at a time — no inter-slice context.

3D approach: the full volume [B,1,4,H,W] is processed jointly — the model
             sees all 4 slices simultaneously and can exploit inter-slice
             context to detect tumours that span multiple slices.

Outputs:
  training_curves.png   — loss / accuracy / IoU per epoch for all 6 models
  test_results.png      — bar chart of final test metrics
  qualitative.png       — sample predictions vs ground truth
  Console report        — declares which dimension wins and by how much
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from DataGen import get_dataloaders, NUM_CLASSES, CLASS_NAMES
from Models  import get_2d_models, get_3d_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LR     = 1e-3
BATCH  = 12   # volume batch size; 2D sees BATCH*N_SLICES slices per step


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────

def pixel_acc(pred_cls, target):
    return (pred_cls == target).float().mean().item()


def mean_iou(pred_cls, target, nc=NUM_CLASSES):
    ious = []
    for c in range(nc):
        inter = ((pred_cls == c) & (target == c)).sum().item()
        union = ((pred_cls == c) | (target == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ──────────────────────────────────────────────────────────
# 2D helpers  —  split volume into individual slices
# ──────────────────────────────────────────────────────────

def _to_slices(volumes, masks):
    """[B,4,H,W] → images [B*4,1,H,W], targets [B*4,H,W]."""
    B, S, H, W = volumes.shape
    return volumes.view(B * S, 1, H, W), masks.view(B * S, H, W)


def _run_2d(model, loader, optimizer, criterion, training):
    model.train(training)
    tot_loss = tot_acc = tot_iou = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for vols, msks in loader:
            imgs, tgts = _to_slices(vols, msks)
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, tgts)
            if training:
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)
            tot_loss += loss.item()
            tot_acc  += pixel_acc(pred, tgts)
            tot_iou  += mean_iou(pred, tgts)
    n = len(loader)
    return tot_loss / n, tot_acc / n, tot_iou / n


# ──────────────────────────────────────────────────────────
# 3D helpers  —  process full volume
# ──────────────────────────────────────────────────────────

def _run_3d(model, loader, optimizer, criterion, training):
    model.train(training)
    tot_loss = tot_acc = tot_iou = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for vols, msks in loader:
            imgs = vols.unsqueeze(1).to(DEVICE)   # [B,1,4,H,W]
            tgts = msks.to(DEVICE)                # [B,4,H,W]
            if training:
                optimizer.zero_grad()
            logits = model(imgs)                   # [B,NC,4,H,W]
            loss   = criterion(logits, tgts)       # PyTorch N-D cross-entropy
            if training:
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)                # [B,4,H,W]
            tot_loss += loss.item()
            tot_acc  += pixel_acc(pred, tgts)
            tot_iou  += mean_iou(pred, tgts)
    n = len(loader)
    return tot_loss / n, tot_acc / n, tot_iou / n


# ──────────────────────────────────────────────────────────
# Generic training loop
# ──────────────────────────────────────────────────────────

def train_model(name, model, train_loader, val_loader, epochs=EPOCHS):
    is_3d     = name.startswith("3D")
    run_fn    = _run_3d if is_3d else _run_2d
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {k: [] for k in
               ("train_loss", "val_loss", "train_acc", "val_acc", "train_iou", "val_iou")}

    print(f"\n{'='*62}\n  {name}  ({'3D volume' if is_3d else '2D per-slice'})\n{'='*62}")
    print(f"  {'Ep':>3}  TrLoss  TrAcc  TrIoU  VlLoss  VlAcc  VlIoU")
    print("  " + "─" * 54)

    for ep in range(1, epochs + 1):
        tr_l, tr_a, tr_i = run_fn(model, train_loader, optimizer, criterion, training=True)
        vl_l, vl_a, vl_i = run_fn(model, val_loader,   optimizer, criterion, training=False)
        scheduler.step()

        history["train_loss"].append(tr_l); history["val_loss"].append(vl_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(vl_a)
        history["train_iou"].append(tr_i);  history["val_iou"].append(vl_i)

        print(f"  {ep:3d}  {tr_l:.4f}  {tr_a:.4f}  {tr_i:.4f}"
              f"  {vl_l:.4f}  {vl_a:.4f}  {vl_i:.4f}")

    return history


# ──────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────

def _color(name):
    return "tab:blue" if name.startswith("2D") else "tab:orange"

def _style(name):
    return "-" if "Small" in name else ("--" if "Medium" in name else ":")


def plot_curves(all_histories, save="training_curves.png"):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("2D vs 3D UNet — Training Curves  (blue=2D, orange=3D)",
                 fontsize=13, fontweight="bold")
    epochs = range(1, len(next(iter(all_histories.values()))["train_loss"]) + 1)

    panels = [
        (axes[0, 0], "train_loss", "Train Loss"),
        (axes[0, 1], "val_loss",   "Val Loss"),
        (axes[0, 2], "train_acc",  "Train Pixel Acc"),
        (axes[1, 0], "val_acc",    "Val Pixel Acc"),
        (axes[1, 1], "train_iou",  "Train Mean IoU"),
        (axes[1, 2], "val_iou",    "Val Mean IoU"),
    ]
    for ax, key, title in panels:
        for name, h in all_histories.items():
            ax.plot(epochs, h[key], label=name,
                    color=_color(name), linestyle=_style(name), linewidth=1.8)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save}")


def plot_test_bars(test_results, save="test_results.png"):
    names   = list(test_results.keys())
    colors  = [_color(n) for n in names]
    metrics = ["loss", "acc", "iou"]
    titles  = ["Test Loss ↓", "Test Pixel Acc ↑", "Test Mean IoU ↑"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Test Set Results — 2D vs 3D UNet", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [test_results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    legend = [mpatches.Patch(facecolor="tab:blue",   label="2D models"),
              mpatches.Patch(facecolor="tab:orange", label="3D models")]
    fig.legend(handles=legend, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save}")


def plot_qualitative(all_models, test_loader, save="qualitative.png"):
    """3 volumes × 2 slices: input | GT | 2D-Small | 2D-Medium | 3D-Small | 3D-Medium."""
    CMAP        = plt.colormaps["tab10"].resampled(NUM_CLASSES)
    show_models = {k: v for k, v in all_models.items()
                   if k in ("2D-Small", "2D-Medium", "3D-Small", "3D-Medium")}
    n_show      = 3
    n_slices    = 2
    n_cols      = 2 + len(show_models)   # input + GT + 4 models

    vols_b, msks_b = next(iter(test_loader))

    fig, axes = plt.subplots(n_show * n_slices, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_show * n_slices))

    col_titles = ["Input", "Ground Truth"] + list(show_models.keys())
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8, fontweight="bold")

    row = 0
    for si in range(n_show):
        vol = vols_b[si]   # [4, H, W]
        gt  = msks_b[si]   # [4, H, W]
        for z in range(n_slices):
            axes[row, 0].imshow(vol[z].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, 0].axis("off")
            axes[row, 1].imshow(gt[z].numpy(), cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
            axes[row, 1].axis("off")

            for col, (name, model) in enumerate(show_models.items(), start=2):
                model.eval()
                with torch.no_grad():
                    if name.startswith("2D"):
                        inp    = vol[z].unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]
                        pred   = model(inp).argmax(1).squeeze(0).cpu().numpy()
                    else:
                        inp    = vol.unsqueeze(0).unsqueeze(0).to(DEVICE)     # [1,1,4,H,W]
                        pred   = model(inp).argmax(1)[0, z].cpu().numpy()
                axes[row, col].imshow(pred, cmap=CMAP, vmin=0, vmax=NUM_CLASSES - 1)
                axes[row, col].axis("off")
            row += 1

    patches = [mpatches.Patch(color=CMAP(c), label=f"{c}: {CLASS_NAMES[c]}")
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Qualitative Comparison  (blue cols = 2D, orange cols = 3D)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved {save}")


# ──────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────

def print_report(all_histories, test_results):
    div = "=" * 66
    print(f"\n{div}")
    print("  FINAL REPORT — 2D vs 3D UNet  Brain Tumour Segmentation")
    print(div)
    print(f"\n  {'Model':<15} {'BestValAcc':>10} {'BestValIoU':>10}"
          f" {'TestLoss':>9} {'TestAcc':>8} {'TestIoU':>8}")
    print("  " + "─" * 62)

    for name, res in test_results.items():
        h = all_histories[name]
        print(f"  {name:<15} {max(h['val_acc']):10.4f} {max(h['val_iou']):10.4f}"
              f" {res['loss']:9.4f} {res['acc']:8.4f} {res['iou']:8.4f}")

    d2_iou = np.mean([test_results[n]["iou"] for n in test_results if n.startswith("2D")])
    d3_iou = np.mean([test_results[n]["iou"] for n in test_results if n.startswith("3D")])
    d2_acc = np.mean([test_results[n]["acc"] for n in test_results if n.startswith("2D")])
    d3_acc = np.mean([test_results[n]["acc"] for n in test_results if n.startswith("3D")])

    winner_iou = "3D" if d3_iou > d2_iou else "2D"
    winner_acc = "3D" if d3_acc > d2_acc else "2D"

    print(f"\n  Group averages (test set):")
    print(f"    2D models — IoU: {d2_iou:.4f}   Acc: {d2_acc:.4f}")
    print(f"    3D models — IoU: {d3_iou:.4f}   Acc: {d3_acc:.4f}")
    print(f"\n  ▶  {winner_iou} models win on mean IoU  (Δ = {abs(d3_iou - d2_iou):.4f})")
    print(f"  ▶  {winner_acc} models win on pixel Acc (Δ = {abs(d3_acc - d2_acc):.4f})")

    if winner_iou == "3D":
        print("\n  Interpretation: 3D models benefit from inter-slice context —")
        print("  they can track tumours that are small or faint in a single slice")
        print("  but consistent across the volume.")
    else:
        print("\n  Interpretation: 2D models match or exceed 3D here — the tumours")
        print("  are large enough to be identifiable per-slice, and 2D models")
        print("  generalise efficiently without the 3D memory overhead.")
    print(f"{div}\n")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_size=2000, val_size=400, test_size=400, batch_size=BATCH
    )

    models_2d = get_2d_models(DEVICE)
    models_3d = get_3d_models(DEVICE)
    criterion = nn.CrossEntropyLoss()

    all_histories = {}
    test_results  = {}

    for name, model in {**models_2d, **models_3d}.items():
        history = train_model(name, model, train_loader, val_loader, epochs=EPOCHS)
        all_histories[name] = history

        is_3d    = name.startswith("3D")
        eval_fn  = _run_3d if is_3d else _run_2d
        tl, ta, ti = eval_fn(model, test_loader, None, criterion, training=False)
        test_results[name] = {"loss": tl, "acc": ta, "iou": ti}
        print(f"  → Test | Loss: {tl:.4f}  Acc: {ta:.4f}  IoU: {ti:.4f}")

    plot_curves(all_histories)
    plot_test_bars(test_results)
    plot_qualitative({**models_2d, **models_3d}, test_loader)
    print_report(all_histories, test_results)


if __name__ == "__main__":
    main()
