from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.optim as optim

from training.engine import TrainingEngine, TrainingConfig, TrainingRunMetrics
from training.metrics import merge_confusion_matrices
from utils import EarlyStopping, CheckpointSaver


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class LoopConfig:
    epochs: int = 30
    gradient_clip: Optional[float] = 1.0
    use_checkpointing: bool = True
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: Optional[int] = 10


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_dice: list = field(default_factory=list)
    val_dice: list = field(default_factory=list)
    train_iou: list = field(default_factory=list)
    val_iou: list = field(default_factory=list)
    train_accuracy: list = field(default_factory=list)
    val_accuracy: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)


@dataclass
class TrainingSummary:
    best_epoch: int
    best_val_metrics: TrainingRunMetrics
    history: TrainingHistory
    total_time_sec: float
    state_dict: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, torch.Tensor]
    confusion_matrix: Optional[torch.Tensor]


class TrainEvalRunner:
    """High level training loop that orchestrates optimisation, validation, and checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        engine_config: TrainingConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
        loop_config: LoopConfig,
        device: torch.device,
        model_name: str,
    ) -> None:
        self.model = model.to(device)
        self.engine_config = engine_config
        self.optimizer = optim.AdamW(model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=scheduler_config.patience,
            factor=scheduler_config.factor,
            min_lr=scheduler_config.min_lr,
        )
        self.loop_config = loop_config
        self.device = device
        self.model_name = model_name

        self.checkpoint_saver: Optional[CheckpointSaver] = None
        if loop_config.use_checkpointing:
            self.checkpoint_saver = CheckpointSaver(checkpoint_dir=loop_config.checkpoint_dir, verbose=False)

        self.early_stopping: Optional[EarlyStopping] = None
        if loop_config.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(patience=loop_config.early_stopping_patience, verbose=False)

        self.engine = TrainingEngine(
            model=self.model,
            optimizer=self.optimizer,
            criterion=None,  # placeholder, set later per run
            device=device,
            config=engine_config,
        )

    def fit(
        self,
        criterion: torch.nn.Module,
        train_loader,
        val_loader,
    ) -> TrainingSummary:
        import time

        self.engine.criterion = criterion

        history = TrainingHistory()
        best_val_metric = None
        best_epoch = -1
        best_state = None
        best_opt_state = None
        aggregate_confusion = None

        start_time = time.time()

        for epoch in range(self.loop_config.epochs):
            train_run = self.engine.train_epoch(train_loader)
            val_run = self.engine.evaluate(val_loader)

            self.scheduler.step(val_run.loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            history.train_loss.append(train_run.loss)
            history.val_loss.append(val_run.loss)
            history.train_dice.append(train_run.dice)
            history.val_dice.append(val_run.dice)
            history.train_iou.append(train_run.iou)
            history.val_iou.append(val_run.iou)
            history.train_accuracy.append(train_run.accuracy)
            history.val_accuracy.append(val_run.accuracy)
            history.learning_rates.append(current_lr)

            aggregate_confusion = merge_confusion_matrices(aggregate_confusion, val_run.confusion_matrix)

            if best_val_metric is None or val_run.dice > best_val_metric.dice:
                best_val_metric = val_run
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                best_opt_state = {k: v for k, v in self.optimizer.state_dict().items()}
                if self.checkpoint_saver:
                    self.checkpoint_saver.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=val_run.loss,
                        metrics={'dice': val_run.dice, 'iou': val_run.iou},
                        model_name=self.model_name,
                    )

            if self.early_stopping is not None:
                self.early_stopping(val_run.loss)
                if self.early_stopping.early_stop:
                    break

        total_time = time.time() - start_time

        summary = TrainingSummary(
            best_epoch=best_epoch,
            best_val_metrics=best_val_metric,
            history=history,
            total_time_sec=total_time,
            state_dict=best_state or self.model.state_dict(),
            optimizer_state=best_opt_state or self.optimizer.state_dict(),
            confusion_matrix=None if aggregate_confusion is None else torch.tensor(aggregate_confusion),
        )
        return summary

    def evaluate(self, criterion: torch.nn.Module, dataloader) -> TrainingRunMetrics:
        self.engine.criterion = criterion
        return self.engine.evaluate(dataloader)
