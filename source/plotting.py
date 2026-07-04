from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: torch.Tensor | Sequence[float]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_polynomial_fit(lambdas: torch.Tensor, target: torch.Tensor, fitted: torch.Tensor, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.0))
    plt.plot(_to_numpy(lambdas), _to_numpy(target), color="#00FF00", linestyle="-", linewidth=2, label="Target")
    plt.plot(_to_numpy(lambdas), _to_numpy(fitted), color="blue", linestyle=":", linewidth=2, label="Fitted")
    plt.title(title)
    plt.xlabel(r"$\lambda$")
    plt.xlim(-1, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_constructive_grid(
    lambdas: torch.Tensor,
    target_by_name: Dict[str, torch.Tensor],
    fitted_by_name_delta: Dict[str, Dict[float, torch.Tensor]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [("lambda", r"$\lambda$"), ("quartic", r"$3(\lambda+0.8)\lambda(\lambda-0.5)^2+0.3$")]
    max_cols = max(len(fitted_by_name_delta[name]) for name, _ in rows)
    fig, axes = plt.subplots(len(rows), max_cols, figsize=(4.2 * max_cols, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for row_idx, (name, _) in enumerate(rows):
        for col_idx, (delta, fitted) in enumerate(fitted_by_name_delta[name].items()):
            ax = axes[row_idx, col_idx]
            ax.plot(_to_numpy(lambdas), _to_numpy(target_by_name[name]), color="#00FF00", linestyle="-", linewidth=2)
            ax.plot(_to_numpy(lambdas), _to_numpy(fitted), color="blue", linestyle=":", linewidth=2)
            ax.set_xlim(-1, 1)
            ax.set_title(rf"$\Delta={delta:g}$")
            ax.set_xlabel(r"$\lambda$")
        for col_idx in range(len(fitted_by_name_delta[name]), max_cols):
            axes[row_idx, col_idx].axis("off")
    handles = [
        plt.Line2D([0], [0], color="#00FF00", linestyle="-", linewidth=2, label="Target"),
        plt.Line2D([0], [0], color="blue", linestyle=":", linewidth=2, label="Fitted"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_accuracy_curves(results_by_depth: Dict[int, Sequence[float]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 4.2))
    for depth, accuracies in results_by_depth.items():
        epochs = np.arange(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, label=rf"$L={depth}$")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.45, 1.01)
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction_histogram(
    metric_values: torch.Tensor,
    predictions: torch.Tensor,
    path: Path,
    xlabel: str,
    title: str,
    bins: int = 50,
    xlim: tuple[float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_np = _to_numpy(metric_values)
    pred_np = _to_numpy(predictions).astype(int)
    plt.figure(figsize=(5.2, 3.6))
    plt.hist(metric_np[pred_np == 0], bins=bins, alpha=0.55, color="red", label="Predicted 0")
    plt.hist(metric_np[pred_np == 1], bins=bins, alpha=0.55, color="blue", label="Predicted 1")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bloch_dataset(bloch_vectors: torch.Tensor, labels: torch.Tensor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bloch_np = _to_numpy(bloch_vectors)
    labels_np = _to_numpy(labels).astype(int)
    fig = plt.figure(figsize=(6.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    class0 = labels_np == 0
    class1 = labels_np == 1
    ax.scatter(bloch_np[class0, 0], bloch_np[class0, 1], bloch_np[class0, 2], c="red", marker="s", s=8, label="Class 0")
    ax.scatter(bloch_np[class1, 0], bloch_np[class1, 1], bloch_np[class1, 2], c="blue", marker="o", s=8, label="Class 1")
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.set_zlabel(r"$r_3$")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
