from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from config import (
    BLOCH_RULE_A_DEPTHS,
    BLOCH_RULE_B_DEPTHS,
    CLASSIFIER_DEPTHS,
    CLASSIFICATION_NUM_EPOCHS,
    CONSTRUCTIVE_DELTAS,
    LEARNING_RATE,
    POLY_GRID_MAX,
    POLY_GRID_MIN,
    POLY_GRID_POINTS,
    POLY_NUM_EPOCHS,
    POLY_TARGETS,
    TEST_SIZE,
    TRAIN_SIZE,
)
from data import (
    make_bloch_sphere_dataset,
    make_entanglement_entropy_dataset,
    make_purity_dataset,
    sample_bloch_sphere,
    set_seed,
)
from models import FullReuploadModel, RestrictedSU2CNOTModel
from plotting import (
    plot_accuracy_curves,
    plot_bloch_dataset,
    plot_constructive_grid,
    plot_polynomial_fit,
    plot_prediction_histogram,
)
from quantum import REAL_DTYPE, single_qubit_density_from_bloch
from training import train_classifier, train_regression


def polynomial_target(name: str, lambdas: torch.Tensor) -> torch.Tensor:
    if name == "lambda":
        return lambdas
    if name == "quartic":
        return 3.0 * (lambdas + 0.8) * lambdas * (lambdas - 0.5) ** 2 + 0.3
    if name == "cos":
        return torch.cos(lambdas)
    raise ValueError(f"Unknown target {name}.")


def constructive_rotation_eval(lambdas: torch.Tensor, coeffs_desc: list[float], delta: float) -> torch.Tensor:
    batch_size = lambdas.shape[0]
    v = torch.zeros(batch_size, 2, dtype=REAL_DTYPE, device=lambdas.device)
    v[:, 0] = 1.0
    for coeff in coeffs_desc[:-1]:
        theta = torch.tensor(delta * coeff, dtype=REAL_DTYPE, device=lambdas.device)
        c = torch.cos(theta)
        s = torch.sin(theta)
        rotation = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        v = v @ rotation.T
        v[:, 1] = v[:, 1] * lambdas
    return v[:, 1] / delta + coeffs_desc[-1]


def run_polynomial_fitting(
    output_dir: Path,
    device: torch.device,
    epochs: int = POLY_NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: Optional[int] = 0,
    make_plots: bool = True,
    print_every: int = 100,
) -> Dict[str, float]:
    print("\nPolynomial fitting")
    set_seed(seed)
    out = output_dir / "polynomial_fitting"
    out.mkdir(parents=True, exist_ok=True)
    lambdas = torch.linspace(POLY_GRID_MIN, POLY_GRID_MAX, steps=POLY_GRID_POINTS, device=device)
    summary: Dict[str, float] = {}

    for name, spec in POLY_TARGETS.items():
        target = polynomial_target(name, lambdas)
        model = RestrictedSU2CNOTModel(num_layers=spec["L"], initial_state="zero").to(device)
        print(f"\nTraining restricted SU(2)+CNOT target={name}, L={spec['L']}, epochs={epochs}, lr={lr}")
        result = train_regression(model, lambdas, target, epochs=epochs, lr=lr, print_every=print_every)
        summary[f"{name}_best_loss"] = result.best_loss
        print(f"Best loss for {name}: {result.best_loss:.10g} at epoch {result.best_epoch}")
        if make_plots:
            plot_polynomial_fit(
                lambdas,
                target,
                result.best_predictions.to(device),
                spec["title"],
                out / f"fit_{name}_L{spec['L']}.png",
            )

    constructive_targets: Dict[str, torch.Tensor] = {}
    constructive_fits: Dict[str, Dict[float, torch.Tensor]] = {}
    for name in ["lambda", "quartic"]:
        coeffs = POLY_TARGETS[name]["coeffs_desc"]
        if coeffs is None:
            continue
        constructive_targets[name] = polynomial_target(name, lambdas)
        constructive_fits[name] = {}
        for delta in CONSTRUCTIVE_DELTAS[name]:
            constructive_fits[name][delta] = constructive_rotation_eval(lambdas, coeffs, delta).detach().cpu()
    if make_plots:
        plot_constructive_grid(lambdas, constructive_targets, constructive_fits, out / "constructive.png")
    return summary


def _run_depth_sweep(
    task_name: str,
    dataset,
    input_dim: int,
    depths: list[int],
    output_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: Optional[int],
    make_plots: bool,
    print_every: int,
    hist_xlabel: str,
    hist_xlim: tuple[float, float] | None,
) -> Dict[int, float]:
    results_by_depth = {}
    accuracies_by_depth = {}
    for depth in depths:
        print(f"\nTraining {task_name}: unrestricted, L={depth}, epochs={epochs}, lr={lr}")
        if seed is not None:
            torch.manual_seed(seed + 10_000 + depth)
        model = FullReuploadModel(input_dim=input_dim, num_layers=depth).to(device)
        result = train_classifier(
            model,
            dataset.rho_train,
            dataset.y_train,
            dataset.rho_test,
            dataset.y_test,
            epochs=epochs,
            lr=lr,
            print_every=print_every,
        )
        results_by_depth[depth] = result.best_test_accuracy
        accuracies_by_depth[depth] = result.test_accuracies
        print(f"Best test accuracy for L={depth}: {result.best_test_accuracy:.4f} at epoch {result.best_epoch}")
        if make_plots:
            safe_task = task_name.lower().replace(" ", "_")
            plot_prediction_histogram(
                dataset.metric_test.detach().cpu(),
                result.best_test_predictions,
                output_dir / f"{safe_task}_hist_L{depth}.png",
                xlabel=hist_xlabel,
                title=rf"{task_name}, $L={depth}$, acc={result.best_test_accuracy:.2f}",
                xlim=hist_xlim,
            )
    if make_plots:
        safe_task = task_name.lower().replace(" ", "_")
        plot_accuracy_curves(
            accuracies_by_depth,
            output_dir / f"{safe_task}_accuracy_curves.png",
            title=task_name,
        )
    return results_by_depth


def run_purity_classification(
    output_dir: Path,
    device: torch.device,
    epochs: int = CLASSIFICATION_NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: Optional[int] = 0,
    make_plots: bool = True,
    print_every: int = 50,
) -> Dict[int, float]:
    print("\nPurity classification")
    set_seed(seed)
    dataset = make_purity_dataset(TRAIN_SIZE, TEST_SIZE, device)
    out = output_dir / "purity_classification"
    out.mkdir(parents=True, exist_ok=True)
    return _run_depth_sweep(
        task_name="Purity classification",
        dataset=dataset,
        input_dim=2,
        depths=CLASSIFIER_DEPTHS,
        output_dir=out,
        device=device,
        epochs=epochs,
        lr=lr,
        seed=seed,
        make_plots=make_plots,
        print_every=print_every,
        hist_xlabel="Purity",
        hist_xlim=(0.5, 1.0),
    )


def run_entanglement_entropy_classification(
    output_dir: Path,
    device: torch.device,
    epochs: int = CLASSIFICATION_NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: Optional[int] = 0,
    make_plots: bool = True,
    print_every: int = 50,
) -> Dict[int, float]:
    print("\nEntanglement entropy classification")
    set_seed(seed)
    dataset = make_entanglement_entropy_dataset(TRAIN_SIZE, TEST_SIZE, device)
    out = output_dir / "entanglement_entropy_classification"
    out.mkdir(parents=True, exist_ok=True)
    return _run_depth_sweep(
        task_name="Entanglement entropy classification",
        dataset=dataset,
        input_dim=4,
        depths=CLASSIFIER_DEPTHS,
        output_dir=out,
        device=device,
        epochs=epochs,
        lr=lr,
        seed=seed,
        make_plots=make_plots,
        print_every=print_every,
        hist_xlabel="Entanglement entropy",
        hist_xlim=(0.0, 1.0),
    )


def run_bloch_sphere_classification(
    output_dir: Path,
    device: torch.device,
    epochs: int = CLASSIFICATION_NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: Optional[int] = 0,
    make_plots: bool = True,
    print_every: int = 50,
) -> Dict[str, Dict[int, float]]:
    print("\nClassification on the Bloch Sphere")
    out = output_dir / "bloch_sphere_classification"
    out.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    dataset_a = make_bloch_sphere_dataset(TRAIN_SIZE, TEST_SIZE, "abs_z_ge_half", device)
    set_seed(None if seed is None else seed + 1)
    dataset_b = make_bloch_sphere_dataset(TRAIN_SIZE, TEST_SIZE, "upper_or_middle_lower", device)

    if make_plots:
        set_seed(None if seed is None else seed + 2)
        viz_a = sample_bloch_sphere(5_000, device)
        labels_a = (torch.abs(viz_a[:, 2]) >= 0.5).to(REAL_DTYPE)
        viz_b = sample_bloch_sphere(5_000, device)
        labels_b = ((viz_b[:, 2] >= 0.5) | ((viz_b[:, 2] >= -0.5) & (viz_b[:, 2] < 0.0))).to(REAL_DTYPE)
        plot_bloch_dataset(viz_a.cpu(), labels_a.cpu(), out / "dataset_abs_r3_ge_0p5.png", r"$|r_3|\geq 0.5$")
        plot_bloch_dataset(viz_b.cpu(), labels_b.cpu(), out / "dataset_piecewise_r3.png", r"$r_3\geq0.5$ or $-0.5\leq r_3<0$")

    results_a = _run_depth_sweep(
        task_name="Bloch sphere rule A",
        dataset=dataset_a,
        input_dim=2,
        depths=BLOCH_RULE_A_DEPTHS,
        output_dir=out,
        device=device,
        epochs=epochs,
        lr=lr,
        seed=seed,
        make_plots=make_plots,
        print_every=print_every,
        hist_xlabel=r"$r_3$",
        hist_xlim=(-1.0, 1.0),
    )
    results_b = _run_depth_sweep(
        task_name="Bloch sphere rule B",
        dataset=dataset_b,
        input_dim=2,
        depths=BLOCH_RULE_B_DEPTHS,
        output_dir=out,
        device=device,
        epochs=epochs,
        lr=lr,
        seed=None if seed is None else seed + 1,
        make_plots=make_plots,
        print_every=print_every,
        hist_xlabel=r"$r_3$",
        hist_xlim=(-1.0, 1.0),
    )
    return {"rule_a": results_a, "rule_b": results_b}
