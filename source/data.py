from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from config import PURITY_THRESHOLD, ENTANGLEMENT_ENTROPY_THRESHOLD
from quantum import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    density_from_statevectors,
    matrix_trace_squared,
    partial_trace_second_qubit_from_two_qubit_statevectors,
    single_qubit_density_from_bloch,
)


@dataclass
class Dataset:
    rho_train: torch.Tensor
    y_train: torch.Tensor
    metric_train: torch.Tensor
    rho_test: torch.Tensor
    y_test: torch.Tensor
    metric_test: torch.Tensor


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=1e-12)


def sample_bloch_ball(num_samples: int, device: torch.device | str) -> torch.Tensor:
    directions = _normalize_rows(torch.randn(num_samples, 3, dtype=REAL_DTYPE, device=device))
    radii = torch.rand(num_samples, 1, dtype=REAL_DTYPE, device=device) ** (1.0 / 3.0)
    return directions * radii


def sample_bloch_sphere(num_samples: int, device: torch.device | str) -> torch.Tensor:
    return _normalize_rows(torch.randn(num_samples, 3, dtype=REAL_DTYPE, device=device))


def haar_statevectors(num_samples: int, dim: int, device: torch.device | str) -> torch.Tensor:
    real = torch.randn(num_samples, dim, dtype=REAL_DTYPE, device=device)
    imag = torch.randn(num_samples, dim, dtype=REAL_DTYPE, device=device)
    psi = (real + 1j * imag).to(COMPLEX_DTYPE)
    return psi / torch.linalg.norm(psi, dim=1, keepdim=True)


def make_purity_dataset(train_size: int, test_size: int, device: torch.device | str) -> Dataset:
    bloch_train = sample_bloch_ball(train_size, device)
    bloch_test = sample_bloch_ball(test_size, device)
    purity_train = 0.5 * (1.0 + torch.sum(bloch_train**2, dim=1))
    purity_test = 0.5 * (1.0 + torch.sum(bloch_test**2, dim=1))
    y_train = (purity_train >= PURITY_THRESHOLD).to(REAL_DTYPE)
    y_test = (purity_test >= PURITY_THRESHOLD).to(REAL_DTYPE)
    return Dataset(
        rho_train=single_qubit_density_from_bloch(bloch_train),
        y_train=y_train,
        metric_train=purity_train,
        rho_test=single_qubit_density_from_bloch(bloch_test),
        y_test=y_test,
        metric_test=purity_test,
    )


def make_entanglement_entropy_dataset(train_size: int, test_size: int, device: torch.device | str) -> Dataset:
    psi_train = haar_statevectors(train_size, dim=4, device=device)
    psi_test = haar_statevectors(test_size, dim=4, device=device)

    reduced_train = partial_trace_second_qubit_from_two_qubit_statevectors(psi_train)
    reduced_test = partial_trace_second_qubit_from_two_qubit_statevectors(psi_test)
    entropy_train = -torch.log2(torch.clamp(matrix_trace_squared(reduced_train), min=1e-12))
    entropy_test = -torch.log2(torch.clamp(matrix_trace_squared(reduced_test), min=1e-12))
    y_train = (entropy_train >= ENTANGLEMENT_ENTROPY_THRESHOLD).to(REAL_DTYPE)
    y_test = (entropy_test >= ENTANGLEMENT_ENTROPY_THRESHOLD).to(REAL_DTYPE)

    return Dataset(
        rho_train=density_from_statevectors(psi_train),
        y_train=y_train,
        metric_train=entropy_train,
        rho_test=density_from_statevectors(psi_test),
        y_test=y_test,
        metric_test=entropy_test,
    )


def make_bloch_sphere_dataset(train_size: int, test_size: int, rule: str, device: torch.device | str) -> Dataset:
    bloch_train = sample_bloch_sphere(train_size, device)
    bloch_test = sample_bloch_sphere(test_size, device)
    z_train = bloch_train[:, 2]
    z_test = bloch_test[:, 2]

    if rule == "abs_z_ge_half":
        y_train = (torch.abs(z_train) >= 0.5).to(REAL_DTYPE)
        y_test = (torch.abs(z_test) >= 0.5).to(REAL_DTYPE)
    elif rule == "upper_or_middle_lower":
        y_train = ((z_train >= 0.5) | ((z_train >= -0.5) & (z_train < 0.0))).to(REAL_DTYPE)
        y_test = ((z_test >= 0.5) | ((z_test >= -0.5) & (z_test < 0.0))).to(REAL_DTYPE)
    else:
        raise ValueError(f"Unknown Bloch-sphere rule: {rule}")

    return Dataset(
        rho_train=single_qubit_density_from_bloch(bloch_train),
        y_train=y_train,
        metric_train=z_train,
        rho_test=single_qubit_density_from_bloch(bloch_test),
        y_test=y_test,
        metric_test=z_test,
    )
