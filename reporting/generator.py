from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from pipeline.config import PipelineConfig


def _frame_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as markdown; fall back to plain text if tabulate is unavailable."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def _history_to_frame(history) -> pd.DataFrame:
    data = {
        "epoch": list(range(1, len(history.train_loss) + 1)),
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "train_dice": history.train_dice,
        "val_dice": history.val_dice,
        "train_iou": history.train_iou,
        "val_iou": history.val_iou,
        "train_accuracy": history.train_accuracy,
        "val_accuracy": history.val_accuracy,
        "lr": history.learning_rates,
    }
    return pd.DataFrame(data)


def _save_history_plot(model_name: str, history, output_dir: Path) -> str:
    df = _history_to_frame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_dice"], label="Train")
    axes[1].plot(df["epoch"], df["val_dice"], label="Val")
    axes[1].set_title("Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["val_iou"], label="Val IoU")
    axes[2].plot(df["epoch"], df["val_accuracy"], label="Val Acc")
    axes[2].set_title("IoU / Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    filename = f"{model_name}_training_curves.png"
    path = output_dir / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return filename


def _save_confusion_matrix(model_name: str, confusion: torch.Tensor | np.ndarray | None, output_dir: Path, class_names: Iterable[str]) -> str | None:
    if confusion is None:
        return None
    matrix = confusion.detach().cpu().numpy() if isinstance(confusion, torch.Tensor) else np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {model_name}")
    filename = f"{model_name}_confusion.png"
    path = output_dir / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return filename


def generate_reports(config: PipelineConfig, results: List) -> None:
    output_dir = Path(config.reporting.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detailed_json = {}

    class_names = ["Background", "Necrotic", "Edema", "Enhancing", "Healthy"][: config.dataset.n_classes]

    for result in results:
        model_name = result.config.name
        summary = result.summary
        test_metrics = result.test_metrics

        history = summary.history
        history_frame = _history_to_frame(history)
        history_csv = output_dir / f"{model_name}_history.csv"
        history_frame.to_csv(history_csv, index=False)

        curve_file = None
        if config.reporting.include_figures:
            curve_file = _save_history_plot(model_name, history, output_dir)
        confusion_file = None
        if config.reporting.include_figures:
            confusion_file = _save_confusion_matrix(model_name, summary.confusion_matrix, output_dir, class_names)

        best_epoch_idx = summary.best_epoch if summary.best_epoch >= 0 else len(history.train_loss) - 1
        best = summary.best_val_metrics
        display_name = result.config.display_name or model_name
        summary_rows.append({
            "model": model_name,
            "display_name": display_name,
            "total_params_m": result.total_params / 1e6,
            "trainable_params_m": result.trainable_params / 1e6,
            "epochs": result.config.loop.epochs,
            "best_epoch": best_epoch_idx + 1,
            "val_dice": best.dice,
            "val_iou": best.iou,
            "val_accuracy": best.accuracy,
            "test_dice": test_metrics.dice,
            "test_iou": test_metrics.iou,
            "test_accuracy": test_metrics.accuracy,
            "training_time_min": summary.total_time_sec / 60.0,
            "curves_file": curve_file,
            "confusion_file": confusion_file,
        })

        detailed_json[model_name] = {
            "config": {
                "optimizer": asdict(result.config.optimizer),
                "scheduler": asdict(result.config.scheduler),
                "loop": asdict(result.config.loop),
                "loss_name": result.config.loss_name,
                "loss_kwargs": result.config.loss_kwargs,
                "extra_model_kwargs": result.config.extra_model_kwargs,
            },
            "history_csv": history_csv.name,
            "training_summary": {
                "best_epoch": best_epoch_idx + 1,
                "best_val_metrics": summary.best_val_metrics.as_dict(),
                "training_time_sec": summary.total_time_sec,
            },
            "evaluation": {
                "validation": summary.best_val_metrics.as_dict(),
                "test": test_metrics.as_dict(),
            },
            "artifacts": {
                "training_curve": curve_file,
                "confusion_matrix": confusion_file,
            },
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    if config.reporting.save_json:
        json_path = output_dir / "summary.json"
        json_path.write_text(json.dumps(detailed_json, indent=2, default=lambda o: o if isinstance(o, (int, float, str)) else None))

    if config.reporting.save_markdown:
        md_path = output_dir / "SUMMARY.md"
        md_lines = ["# Brain Tumor Segmentation Benchmark", ""]
        md_lines.append("## Dataset Configuration")
        md_lines.append(f"- Train samples: {config.dataset.train_samples}")
        md_lines.append(f"- Validation samples: {config.dataset.val_samples}")
        md_lines.append(f"- Test samples: {config.dataset.test_samples}")
        md_lines.append(f"- Image size: {config.dataset.image_size}x{config.dataset.image_size}")
        md_lines.append(f"- Channels: {config.dataset.n_channels}")
        md_lines.append("")
        md_lines.append("## Model Comparison")
        md_lines.append(_frame_to_markdown(summary_df))
        md_lines.append("")
        if config.reporting.include_figures:
            md_lines.append("## Artifacts")
            for row in summary_rows:
                label = row["display_name"]
                if row.get("curves_file"):
                    md_lines.append(f"- **{label}** training curves: {row['curves_file']}")
                if row.get("confusion_file"):
                    md_lines.append(f"  Confusion matrix: {row['confusion_file']}")
            md_lines.append("")
        md_path.write_text("\n".join(md_lines))
