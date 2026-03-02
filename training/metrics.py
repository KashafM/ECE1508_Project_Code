from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Optional, Tuple


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Binary Dice coefficient between two segmentation masks."""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return float(dice.item()) if torch.is_tensor(dice) else float(dice)


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Intersection over Union (IoU) score for binary masks."""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(iou.item()) if torch.is_tensor(iou) else float(iou)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pixel-wise accuracy for multi-class segmentation outputs."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    total = target.numel()
    return float((correct / total).item())


def multi_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    exclude_background: bool = True,
) -> Dict[str, float]:
    """Per-class Dice along with the mean score."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    dice_scores: Dict[str, float] = {}
    start_class = 1 if exclude_background else 0

    for class_idx in range(start_class, num_classes):
        pred_binary = (pred == class_idx).float()
        target_binary = (target == class_idx).float()
        if target_binary.sum() > 0:
            dice_scores[f'class_{class_idx}'] = dice_coefficient(pred_binary, target_binary)

    dice_scores['mean_dice'] = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
    return dice_scores


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_per_class: bool = True,
) -> Dict[str, object]:
    """Aggregate Dice, IoU, accuracy, and optionally per-class statistics."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    metrics: Dict[str, object] = {}
    metrics['accuracy'] = pixel_accuracy(pred, target)

    dice_scores = np.zeros(num_classes, dtype=np.float64)
    iou_scores = np.zeros(num_classes, dtype=np.float64)

    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        dice = (2 * intersection / (union + 1e-7)).item()
        iou = (intersection / (union - intersection + 1e-7)).item()

        dice_scores[class_idx] = dice
        iou_scores[class_idx] = iou

    metrics['dice'] = float(np.mean(dice_scores))
    metrics['iou'] = float(np.mean(iou_scores))

    if include_per_class:
        metrics['per_class_dice'] = dice_scores
        metrics['per_class_iou'] = iou_scores

    return metrics


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """Compute confusion matrix over flattened predictions."""
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target_flat, pred_flat):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1
    return matrix


def merge_confusion_matrices(existing: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
    if existing is None:
        return new.copy()
    return existing + new


class MetricsTracker:
    """Utility for accumulating scalar or vector metrics over multiple batches."""

    def __init__(self) -> None:
        self.metrics: Dict[str, object] = {}
        self.counts: Dict[str, float] = {}

    def reset(self) -> None:
        self.metrics.clear()
        self.counts.clear()

    def update(self, metric_name: str, value: object, count: float = 1.0) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            value = value.astype(np.float64)
            if metric_name not in self.metrics:
                self.metrics[metric_name] = np.zeros_like(value, dtype=np.float64)
                self.counts[metric_name] = 0.0
            self.metrics[metric_name] += value * count
            self.counts[metric_name] += count
        else:
            value = float(value)
            self.metrics[metric_name] = self.metrics.get(metric_name, 0.0) + value * count
            self.counts[metric_name] = self.counts.get(metric_name, 0.0) + count

    def get_average(self, metric_name: str) -> object:
        if metric_name not in self.metrics:
            return 0.0
        total = self.metrics[metric_name]
        count = max(self.counts.get(metric_name, 0.0), 1.0)
        if isinstance(total, np.ndarray):
            return total / count
        return total / count

    def get_all_averages(self) -> Dict[str, object]:
        return {name: self.get_average(name) for name in self.metrics.keys()}
