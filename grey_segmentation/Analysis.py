import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from DataGen import get_dataloaders, NUM_CLASSES, GREY_SHADES
from Models import get_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 16
CLASS_NAMES = ["Background"] + [f"Grey-{v}" for v in GREY_SHADES]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pixel_accuracy(logits, masks):
    """Fraction of pixels correctly classified."""
    predicted = logits.argmax(dim=1)
    return (predicted == masks).float().mean().item()


def mean_iou(logits, masks, num_classes=NUM_CLASSES):
    """Mean Intersection-over-Union across all classes present in the batch."""
    predicted = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = predicted == cls
        true_cls = masks == cls
        intersection = (pred_cls & true_cls).sum().item()
        union = (pred_cls | true_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = total_acc = total_iou = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += pixel_accuracy(logits, masks)
        total_iou += mean_iou(logits, masks)
    n = len(loader)
    return total_loss / n, total_acc / n, total_iou / n


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = total_acc = total_iou = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            total_acc += pixel_accuracy(logits, masks)
            total_iou += mean_iou(logits, masks)
    n = len(loader)
    return total_loss / n, total_acc / n, total_iou / n


# ---------------------------------------------------------------------------
# Training loop for one model
# ---------------------------------------------------------------------------

def train_model(name, model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_iou":  [], "val_iou":  [],
    }

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")
    header = f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrIoU':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'VlIoU':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_iou = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_iou = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(vl_iou)

        print(f"  {epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.4f}  {tr_iou:7.4f}"
              f"  {vl_loss:8.4f}  {vl_acc:7.4f}  {vl_iou:7.4f}")

    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(all_histories, save_path="training_curves.png"):
    """4-panel plot: train/val loss and accuracy per model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("UNet Model Comparison – Grey Circle Segmentation", fontsize=14, fontweight="bold")
    colors = ["tab:blue", "tab:orange", "tab:green"]
    names = list(all_histories.keys())
    epochs = range(1, len(next(iter(all_histories.values()))["train_loss"]) + 1)

    panels = [
        (axes[0, 0], "train_loss", "Training Loss",       "Loss"),
        (axes[0, 1], "val_loss",   "Validation Loss",     "Loss"),
        (axes[1, 0], "train_acc",  "Training Pixel Acc",  "Accuracy"),
        (axes[1, 1], "val_acc",    "Validation Pixel Acc","Accuracy"),
    ]

    for ax, key, title, ylabel in panels:
        for name, color in zip(names, colors):
            ax.plot(epochs, all_histories[name][key], label=name, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


def plot_test_results(test_results, save_path="test_results.png"):
    """Bar charts for test loss, accuracy, and mean IoU."""
    names = list(test_results.keys())
    colors = ["tab:blue", "tab:orange", "tab:green"]
    metrics = ["loss", "acc", "iou"]
    titles  = ["Test Loss", "Test Pixel Accuracy", "Test Mean IoU"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Test Set Results", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [test_results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


def plot_iou_curves(all_histories, save_path="iou_curves.png"):
    """Train and val mean-IoU curves per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Mean IoU Over Training", fontsize=13, fontweight="bold")
    colors = ["tab:blue", "tab:orange", "tab:green"]
    names = list(all_histories.keys())
    epochs = range(1, len(next(iter(all_histories.values()))["train_iou"]) + 1)

    for ax, phase in zip(axes, ["train", "val"]):
        for name, color in zip(names, colors):
            ax.plot(epochs, all_histories[name][f"{phase}_iou"],
                    label=name, color=color, linewidth=2)
        ax.set_title(f"{'Training' if phase == 'train' else 'Validation'} Mean IoU")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean IoU")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


def plot_qualitative(models_dict, test_loader, save_path="qualitative.png"):
    """Show 3 test images with GT mask and each model's prediction."""
    images_batch, masks_batch = next(iter(test_loader))
    images_batch = images_batch[:3]
    masks_batch = masks_batch[:3]

    n_images = images_batch.shape[0]
    n_models = len(models_dict)
    cmap = plt.colormaps["tab10"].resampled(NUM_CLASSES)

    fig, axes = plt.subplots(n_images, 2 + n_models,
                             figsize=(3 * (2 + n_models), 3 * n_images))
    fig.suptitle("Qualitative Results (Input | GT | Model Predictions)", fontsize=12)

    model_names = list(models_dict.keys())

    for row in range(n_images):
        img = images_batch[row]       # [1, H, W]
        gt  = masks_batch[row]        # [H, W]

        axes[row, 0].imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Input" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt.numpy(), cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis("off")

        for col, (name, model) in enumerate(models_dict.items(), start=2):
            model.eval()
            with torch.no_grad():
                logits = model(img.unsqueeze(0).to(DEVICE))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            axes[row, col].imshow(pred, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[row, col].set_title(name if row == 0 else "")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(all_histories, test_results):
    divider = "=" * 65
    print(f"\n{divider}")
    print("  FINAL REPORT – Grey Circle Segmentation")
    print(divider)

    for name, res in test_results.items():
        h = all_histories[name]
        best_val_acc = max(h["val_acc"])
        best_val_iou = max(h["val_iou"])
        print(f"\n  {name}")
        print(f"    Best Val Acc : {best_val_acc:.4f}")
        print(f"    Best Val IoU : {best_val_iou:.4f}")
        print(f"    Test Loss    : {res['loss']:.4f}")
        print(f"    Test Acc     : {res['acc']:.4f}")
        print(f"    Test IoU     : {res['iou']:.4f}")

    winner_acc = max(test_results, key=lambda n: test_results[n]["acc"])
    winner_iou = max(test_results, key=lambda n: test_results[n]["iou"])
    print(f"\n  Best test accuracy : {winner_acc}")
    print(f"  Best test mean IoU : {winner_iou}")
    print(f"{divider}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_size=2000, val_size=400, test_size=400, batch_size=BATCH_SIZE
    )

    models = get_models(DEVICE)
    criterion = nn.CrossEntropyLoss()

    all_histories = {}
    test_results = {}

    for name, model in models.items():
        history = train_model(name, model, train_loader, val_loader, epochs=EPOCHS)
        all_histories[name] = history

        test_loss, test_acc, test_iou = eval_epoch(model, test_loader, criterion)
        test_results[name] = {"loss": test_loss, "acc": test_acc, "iou": test_iou}
        print(f"  → Test | Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  IoU: {test_iou:.4f}")

    # --- plots ---
    plot_training_curves(all_histories)
    plot_iou_curves(all_histories)
    plot_test_results(test_results)
    plot_qualitative(models, test_loader)

    # --- text report ---
    print_report(all_histories, test_results)


if __name__ == "__main__":
    main()
