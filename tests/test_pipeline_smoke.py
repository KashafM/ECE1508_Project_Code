from dataclasses import replace

from dataset_refactored import BrainTumorDataset
from pipeline.config import (
    PipelineConfig,
    ModelRunConfig,
    OptimizerConfig,
    SchedulerConfig,
    LoopConfig,
)
from pipeline.run_pipeline import execute_pipeline
from reporting.generator import generate_reports


def _build_small_config(tmp_path):
    cfg = PipelineConfig.default()
    cfg.device = "cpu"
    cfg.dataset = replace(
        cfg.dataset,
        train_samples=4,
        val_samples=2,
        test_samples=2,
        batch_size=1,
        image_size=128,
        cache_dir=str(tmp_path / "dataset_cache"),
        regenerate_cache=True,
    )

    base_model = cfg.models[0]
    cfg.models = [
        ModelRunConfig(
            name=base_model.name,
            display_name=base_model.display_name,
            optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.0),
            scheduler=SchedulerConfig(patience=1, factor=0.5, min_lr=1e-6),
            loop=LoopConfig(
                epochs=1,
                gradient_clip=1.0,
                use_checkpointing=False,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                early_stopping_patience=1,
            ),
            loss_name="combined",
            loss_kwargs={"ce_weight": 1.0, "dice_weight": 1.0},
            extra_model_kwargs=base_model.extra_model_kwargs,
        )
    ]

    cfg.reporting = replace(
        cfg.reporting,
        output_dir=str(tmp_path / "reports"),
        include_figures=False,
        save_json=True,
        save_markdown=False,
    )
    return cfg


def test_pipeline_execution_smoke(tmp_path):
    cfg = _build_small_config(tmp_path)
    results = execute_pipeline(cfg)
    assert len(results) == 1
    result = results[0]
    assert result.summary.best_val_metrics is not None
    assert result.test_metrics is not None
    generate_reports(cfg, results)
    summary_csv = tmp_path / "reports" / "summary.csv"
    assert summary_csv.exists()
    with summary_csv.open() as f:
        header = f.readline()
    assert "test_dice" in header


def test_dataset_cache_roundtrip(tmp_path):
    cache_dir = tmp_path / "cache"
    dataset = BrainTumorDataset(
        num_samples=3,
        image_size=(64, 64),
        n_channels=2,
        n_classes=5,
        mode='train',
        augment=False,
        normalize=True,
        seed=42,
        cache_dir=str(cache_dir),
        regenerate_cache=True,
    )
    cache_path = dataset.cache_path
    assert cache_path is not None and cache_path.exists()
    assert len(dataset.data_cache) == 3

    # Second instantiation should load from cache without regeneration.
    dataset_cached = BrainTumorDataset(
        num_samples=3,
        image_size=(64, 64),
        n_channels=2,
        n_classes=5,
        mode='train',
        augment=False,
        normalize=True,
        seed=42,
        cache_dir=str(cache_dir),
        regenerate_cache=False,
    )
    assert cache_path.exists()
    assert len(dataset_cached.data_cache) == 3
