from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from training.runner import OptimizerConfig, SchedulerConfig, LoopConfig


@dataclass
class DatasetConfig:
    train_samples: int = 10
    val_samples: int = 2
    test_samples: int = 2
    image_size: int = 256
    n_channels: int = 4
    n_classes: int = 5
    batch_size: int = 4
    num_workers: int = 0
    seed: Optional[int] = 42
    cache_dir: str = "dataset_cache"
    regenerate_cache: bool = False


@dataclass
class ModelRunConfig:
    name: str
    display_name: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loop: LoopConfig = field(default_factory=lambda: LoopConfig(epochs=30))
    loss_name: str = "combined"
    loss_kwargs: dict = field(default_factory=lambda: {"ce_weight": 1.0, "dice_weight": 1.0})
    extra_model_kwargs: dict = field(default_factory=dict)


@dataclass
class ReportingConfig:
    output_dir: str = "reports"
    include_figures: bool = True
    save_json: bool = True
    save_markdown: bool = True


@dataclass
class PipelineConfig:
    device: str = "cuda"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: List[ModelRunConfig] = field(default_factory=list)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @staticmethod
    def default() -> "PipelineConfig":
        base_models = [
            ModelRunConfig(
                name="basic_unet",
                display_name="Basic U-Net",
                loop=LoopConfig(epochs=10, early_stopping_patience=5, gradient_clip=1.0),
            ),
            ModelRunConfig(
                name="standard_unet",
                display_name="Standard U-Net",
                optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-4),
                loop=LoopConfig(epochs=20, early_stopping_patience=7, gradient_clip=1.0),
            ),
            ModelRunConfig(
                name="advanced_unet",
                display_name="Advanced Attention U-Net",
                optimizer=OptimizerConfig(lr=5e-4, weight_decay=1e-4),
                scheduler=SchedulerConfig(patience=7, factor=0.5, min_lr=1e-6),
                loop=LoopConfig(epochs=25, early_stopping_patience=10, gradient_clip=1.0),
                extra_model_kwargs={"use_deep_supervision": True},
            ),
        ]
        return PipelineConfig(models=base_models)
