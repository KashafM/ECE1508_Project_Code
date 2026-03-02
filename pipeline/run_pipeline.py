from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from dataset_refactored import create_dataloaders
from losses import get_loss_function
from models import get_model
from pipeline.config import PipelineConfig, ModelRunConfig, OptimizerConfig, SchedulerConfig, LoopConfig
from reporting.generator import generate_reports
from training.engine import TrainingConfig
from training.runner import TrainEvalRunner


@dataclass
class ModelPipelineResult:
    config: ModelRunConfig
    summary: object
    test_metrics: object
    total_params: int
    trainable_params: int


def load_config(path: Optional[str]) -> PipelineConfig:
    if path is None:
        return PipelineConfig.default()
    data = json.loads(Path(path).read_text())
    default_cfg = PipelineConfig.default()
    dataset_defaults = asdict(default_cfg.dataset)
    reporting_defaults = asdict(default_cfg.reporting)
    dataset = default_cfg.dataset.__class__(**dataset_defaults)
    reporting = default_cfg.reporting.__class__(**reporting_defaults)
    def _parse_model(model_dict: dict) -> ModelRunConfig:
        base = {k: v for k, v in model_dict.items() if k not in {"optimizer", "scheduler", "loop", "loss_kwargs"}}
        optimizer = OptimizerConfig(**model_dict.get("optimizer", {}))
        scheduler = SchedulerConfig(**model_dict.get("scheduler", {}))
        loop = LoopConfig(**model_dict.get("loop", {}))
        loss_kwargs = model_dict.get("loss_kwargs", {})
        return ModelRunConfig(
            optimizer=optimizer,
            scheduler=scheduler,
            loop=loop,
            loss_kwargs=loss_kwargs,
            **base,
        )

    models = [_parse_model(model) for model in data.get("models", [])]
    cfg = PipelineConfig(
        device=data.get("device", default_cfg.device),
        dataset=dataset,
        models=models if models else copy.deepcopy(default_cfg.models),
        reporting=reporting,
    )
    if "dataset" in data:
        cfg.dataset = dataset.__class__(**{**asdict(dataset), **data["dataset"]})
    if "reporting" in data:
        cfg.reporting = reporting.__class__(**{**asdict(reporting), **data["reporting"]})
    return cfg


def execute_pipeline(cfg: PipelineConfig) -> List[ModelPipelineResult]:
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    ds_cfg = cfg.dataset
    dataloaders = create_dataloaders(
        batch_size=ds_cfg.batch_size,
        train_samples=ds_cfg.train_samples,
        val_samples=ds_cfg.val_samples,
        test_samples=ds_cfg.test_samples,
        image_size=(ds_cfg.image_size, ds_cfg.image_size),
        n_channels=ds_cfg.n_channels,
        n_classes=ds_cfg.n_classes,
        num_workers=ds_cfg.num_workers,
        seed=ds_cfg.seed,
        cache_dir=ds_cfg.cache_dir,
        regenerate_cache=ds_cfg.regenerate_cache,
    )

    results: List[ModelPipelineResult] = []
    output_dir = Path(cfg.reporting.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in cfg.models:
        print(f"\n=== Training {model_cfg.name} ===")
        model = get_model(model_cfg.name, **model_cfg.extra_model_kwargs)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        engine_config = TrainingConfig(
            num_classes=ds_cfg.n_classes,
            gradient_clip=model_cfg.loop.gradient_clip,
            deep_supervision=model_cfg.extra_model_kwargs.get("use_deep_supervision", False),
        )

        runner = TrainEvalRunner(
            model=model,
            engine_config=engine_config,
            optimizer_config=model_cfg.optimizer,
            scheduler_config=model_cfg.scheduler,
            loop_config=model_cfg.loop,
            device=device,
            model_name=model_cfg.name,
        )

        criterion = get_loss_function(model_cfg.loss_name, **model_cfg.loss_kwargs)
        summary = runner.fit(criterion, dataloaders['train'], dataloaders['val'])

        model.load_state_dict(summary.state_dict)
        test_metrics = runner.evaluate(criterion, dataloaders['test'])

        results.append(
            ModelPipelineResult(
                config=model_cfg,
                summary=summary,
                test_metrics=test_metrics,
                total_params=total_params,
                trainable_params=trainable_params,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified brain tumour segmentation pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config overriding defaults")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = execute_pipeline(cfg)
    generate_reports(cfg, results)
    print(f"\nReports written to {cfg.reporting.output_dir}")


if __name__ == "__main__":
    main()
