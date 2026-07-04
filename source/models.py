from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from quantum import (
    COMPLEX_DTYPE,
    ket0_dm,
    ketplus_dm,
    reupload_layer,
    restricted_su2_cnot_layer,
    single_parameter_input_density,
    bloch_vector,
)


def hermitian_from_real_params(params: torch.Tensor, matrix_dim: int) -> torch.Tensor:
    if params.numel() != matrix_dim * matrix_dim:
        raise ValueError(f"Expected {matrix_dim * matrix_dim} parameters, got {params.numel()}.")
    h = torch.zeros(matrix_dim, matrix_dim, dtype=COMPLEX_DTYPE, device=params.device)
    idx = 0
    for row in range(matrix_dim):
        h[row, row] = params[idx].to(COMPLEX_DTYPE)
        idx += 1
    for row in range(matrix_dim):
        for col in range(row + 1, matrix_dim):
            real = params[idx]
            imag = params[idx + 1]
            h[row, col] = real.to(COMPLEX_DTYPE) + 1j * imag.to(COMPLEX_DTYPE)
            h[col, row] = real.to(COMPLEX_DTYPE) - 1j * imag.to(COMPLEX_DTYPE)
            idx += 2
    return h


class FullReuploadModel(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.joint_dim = 2 * input_dim
        self.params = nn.Parameter(torch.randn(num_layers, self.joint_dim * self.joint_dim))
        self.w = nn.Parameter(torch.randn(3))
        self.b = nn.Parameter(torch.randn(1))

    def unitaries(self) -> List[torch.Tensor]:
        us: List[torch.Tensor] = []
        for layer in range(self.num_layers):
            h = hermitian_from_real_params(self.params[layer], self.joint_dim)
            us.append(torch.linalg.matrix_exp(1j * h))
        return us

    def forward(self, rho_b: torch.Tensor) -> torch.Tensor:
        batch_size = rho_b.shape[0]
        tau_a = ket0_dm(batch_size, device=rho_b.device)
        for u in self.unitaries():
            tau_a = reupload_layer(tau_a, rho_b, u)
        bloch = bloch_vector(tau_a)
        return bloch @ self.w + self.b.squeeze(0)


class RestrictedSU2CNOTModel(nn.Module):
    def __init__(self, num_layers: int, initial_state: str = "zero"):
        super().__init__()
        if initial_state not in {"zero", "plus"}:
            raise ValueError("initial_state must be 'zero' or 'plus'.")
        self.num_layers = num_layers
        self.initial_state = initial_state
        self.params = nn.Parameter(torch.randn(num_layers, 4))
        self.w = nn.Parameter(torch.randn(3))
        self.b = nn.Parameter(torch.randn(1))

    def unitaries(self) -> List[torch.Tensor]:
        us: List[torch.Tensor] = []
        for layer in range(self.num_layers):
            h = hermitian_from_real_params(self.params[layer], 2)
            us.append(torch.linalg.matrix_exp(1j * h))
        return us

    def forward(self, lambdas: torch.Tensor) -> torch.Tensor:
        rho_b = single_parameter_input_density(lambdas)
        batch_size = rho_b.shape[0]
        if self.initial_state == "plus":
            tau_a = ketplus_dm(batch_size, device=rho_b.device)
        else:
            tau_a = ket0_dm(batch_size, device=rho_b.device)
        for u in self.unitaries():
            tau_a = restricted_su2_cnot_layer(tau_a, rho_b, u)
        bloch = bloch_vector(tau_a)
        return bloch @ self.w + self.b.squeeze(0)
