from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .metrics import MetricsTracker, compute_segmentation_metrics, compute_confusion_matrix, merge_confusion_matrices


@dataclass
class TrainingRunMetrics:
    """Container for the key statistics produced by a training/eval epoch."""

    loss: float
    dice: float
    iou: float
    accuracy: float
    per_class_dice: Optional[np.ndarray] = None
    per_class_iou: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, Union[float, np.ndarray]]:
        data: Dict[str, Union[float, np.ndarray]] = {
            'loss': self.loss,
            'dice': self.dice,
            'iou': self.iou,
            'accuracy': self.accuracy,
        }
        if self.per_class_dice is not None:
            data['per_class_dice'] = (
                self.per_class_dice.tolist()
                if isinstance(self.per_class_dice, np.ndarray)
                else self.per_class_dice
            )
        if self.per_class_iou is not None:
            data['per_class_iou'] = (
                self.per_class_iou.tolist()
                if isinstance(self.per_class_iou, np.ndarray)
                else self.per_class_iou
            )
        if self.confusion_matrix is not None:
            if isinstance(self.confusion_matrix, np.ndarray):
                data['confusion_matrix'] = self.confusion_matrix.tolist()
            elif torch.is_tensor(self.confusion_matrix):
                data['confusion_matrix'] = self.confusion_matrix.cpu().numpy().tolist()
            else:
                data['confusion_matrix'] = self.confusion_matrix
        return data


@dataclass
class TrainingConfig:
    """Configuration options that control the shared training engine."""

    num_classes: int = 5
    gradient_clip: Optional[float] = None
    deep_supervision: bool = False
    deep_supervision_weights: Optional[Sequence[float]] = None
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None


BatchType = Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]


class TrainingEngine:
    """Shared training loop that standardises optimisation and evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: Union[str, torch.device],
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.config = config or TrainingConfig()

        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: Iterable[BatchType], max_batches: Optional[int] = None) -> TrainingRunMetrics:
        limit = max_batches if max_batches is not None else self.config.max_train_batches
        return self._run_epoch(dataloader, train=True, max_batches=limit)

    def evaluate(self, dataloader: Iterable[BatchType], max_batches: Optional[int] = None) -> TrainingRunMetrics:
        limit = max_batches if max_batches is not None else self.config.max_eval_batches
        return self._run_epoch(dataloader, train=False, max_batches=limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_epoch(self, dataloader: Iterable[BatchType], train: bool, max_batches: Optional[int]) -> TrainingRunMetrics:
        tracker = MetricsTracker()
        batch_limit = max_batches if max_batches is not None else float('inf')

        if train:
            self.model.train()
            context = torch.enable_grad()
        else:
            self.model.eval()
            context = torch.no_grad()

        confusion = None
        with context:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= batch_limit:
                    break

                images, masks = self._extract_batch(batch)
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.config.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()

                logits = self._get_logits(outputs)
                metrics = compute_segmentation_metrics(logits, masks, self.config.num_classes)
                confusion = merge_confusion_matrices(
                    confusion,
                    compute_confusion_matrix(logits, masks, self.config.num_classes),
                )

                tracker.update('loss', float(loss.item()))
                tracker.update('dice', metrics['dice'])
                tracker.update('iou', metrics['iou'])
                tracker.update('accuracy', metrics['accuracy'])

                if 'per_class_dice' in metrics:
                    tracker.update('per_class_dice', metrics['per_class_dice'])
                if 'per_class_iou' in metrics:
                    tracker.update('per_class_iou', metrics['per_class_iou'])

        averages = tracker.get_all_averages()
        return TrainingRunMetrics(
            loss=float(averages.get('loss', 0.0)),
            dice=float(averages.get('dice', 0.0)),
            iou=float(averages.get('iou', 0.0)),
            accuracy=float(averages.get('accuracy', 0.0)),
            per_class_dice=averages.get('per_class_dice'),
            per_class_iou=averages.get('per_class_iou'),
            confusion_matrix=confusion,
        )

    def _compute_loss(self, outputs: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            if self.config.deep_supervision:
                weights = self._normalise_deep_supervision_weights(len(outputs))
                total_loss: Optional[torch.Tensor] = None
                for output, weight in zip(outputs, weights):
                    term = self.criterion(output, target)
                    total_loss = term * weight if total_loss is None else total_loss + weight * term
                return total_loss if total_loss is not None else self.criterion(outputs[-1], target)
            outputs = outputs[-1]
        return self.criterion(outputs, target)

    def _normalise_deep_supervision_weights(self, depth: int) -> Sequence[float]:
        weights = self.config.deep_supervision_weights
        if weights is None or len(weights) == 0:
            return [1.0] * depth
        if len(weights) >= depth:
            return weights[:depth]
        last_weight = weights[-1]
        return list(weights) + [last_weight] * (depth - len(weights))

    @staticmethod
    def _get_logits(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            return outputs[-1]
        return outputs

    @staticmethod
    def _extract_batch(batch: BatchType) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch['image'], batch['mask']
        return batch
