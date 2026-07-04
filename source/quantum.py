from __future__ import annotations

from typing import Tuple

import torch

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


def paulis(device: torch.device | str | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {"dtype": COMPLEX_DTYPE, "device": device}
    eye = torch.eye(2, **kwargs)
    x = torch.tensor([[0, 1], [1, 0]], **kwargs)
    y = torch.tensor([[0, -1j], [1j, 0]], **kwargs)
    z = torch.tensor([[1, 0], [0, -1]], **kwargs)
    return eye, x, y, z


def ket(index: int, dim: int, device: torch.device | str | None = None) -> torch.Tensor:
    out = torch.zeros(dim, 1, dtype=COMPLEX_DTYPE, device=device)
    out[index, 0] = 1.0 + 0.0j
    return out


def ket0_dm(batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    k0 = ket(0, 2, device)
    dm = k0 @ k0.conj().T
    return dm.unsqueeze(0).expand(batch_size, 2, 2).clone()


def ketplus_dm(batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    k0 = ket(0, 2, device)
    k1 = ket(1, 2, device)
    kp = (k0 + k1) / torch.sqrt(torch.tensor(2.0, dtype=REAL_DTYPE, device=device))
    dm = kp @ kp.conj().T
    return dm.unsqueeze(0).expand(batch_size, 2, 2).clone()


def single_qubit_density_from_bloch(bloch: torch.Tensor) -> torch.Tensor:
    bloch = bloch.to(dtype=REAL_DTYPE)
    device = bloch.device
    eye, x, y, z = paulis(device)
    return 0.5 * (
        eye
        + bloch[..., 0, None, None].to(COMPLEX_DTYPE) * x
        + bloch[..., 1, None, None].to(COMPLEX_DTYPE) * y
        + bloch[..., 2, None, None].to(COMPLEX_DTYPE) * z
    )


def single_parameter_input_density(lambdas: torch.Tensor) -> torch.Tensor:
    lambdas = lambdas.to(dtype=REAL_DTYPE)
    device = lambdas.device
    eye, x, _, z = paulis(device)
    x_coeff = torch.sqrt(torch.clamp(1.0 - lambdas**2, min=0.0))
    return 0.5 * (
        eye.unsqueeze(0)
        + x_coeff[:, None, None].to(COMPLEX_DTYPE) * x.unsqueeze(0)
        + lambdas[:, None, None].to(COMPLEX_DTYPE) * z.unsqueeze(0)
    )


def density_from_statevectors(statevectors: torch.Tensor) -> torch.Tensor:
    statevectors = statevectors.to(dtype=COMPLEX_DTYPE)
    return statevectors[:, :, None] @ statevectors[:, None, :].conj()


def batched_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, da, _ = a.shape
    _, db, _ = b.shape
    return torch.einsum("nij,nkl->nikjl", a, b).reshape(n, da * db, da * db)


def partial_trace_second(rho_ab: torch.Tensor, dim_a: int, dim_b: int) -> torch.Tensor:
    n = rho_ab.shape[0]
    reshaped = rho_ab.reshape(n, dim_a, dim_b, dim_a, dim_b)
    return torch.einsum("nikjk->nij", reshaped)


def reupload_layer(tau_a: torch.Tensor, rho_b: torch.Tensor, unitary_ab: torch.Tensor) -> torch.Tensor:
    dim_b = rho_b.shape[-1]
    joint = batched_kron(tau_a, rho_b)
    u = unitary_ab
    joint = torch.matmul(u.unsqueeze(0), joint)
    joint = torch.matmul(joint, u.conj().T.unsqueeze(0))
    return partial_trace_second(joint, dim_a=2, dim_b=dim_b)


def cnot_b_to_a(device: torch.device | str | None = None) -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=COMPLEX_DTYPE,
        device=device,
    )


def restricted_su2_cnot_layer(tau_a: torch.Tensor, rho_b: torch.Tensor, unitary_a: torch.Tensor) -> torch.Tensor:
    tau_a = torch.matmul(unitary_a.unsqueeze(0), tau_a)
    tau_a = torch.matmul(tau_a, unitary_a.conj().T.unsqueeze(0))
    joint = batched_kron(tau_a, rho_b)
    cnot = cnot_b_to_a(device=tau_a.device)
    joint = torch.matmul(cnot.unsqueeze(0), joint)
    joint = torch.matmul(joint, cnot.conj().T.unsqueeze(0))
    return partial_trace_second(joint, dim_a=2, dim_b=2)


def bloch_vector(tau_a: torch.Tensor) -> torch.Tensor:
    _, x, y, z = paulis(tau_a.device)
    bx = torch.einsum("nij,ji->n", tau_a, x).real
    by = torch.einsum("nij,ji->n", tau_a, y).real
    bz = torch.einsum("nij,ji->n", tau_a, z).real
    return torch.stack([bx, by, bz], dim=-1)


def partial_trace_second_qubit_from_two_qubit_statevectors(psi: torch.Tensor) -> torch.Tensor:
    coeffs = psi.reshape(psi.shape[0], 2, 2)
    return torch.einsum("naj,nbj->nab", coeffs, coeffs.conj())


def matrix_trace_squared(dm: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nij,nji->n", dm, dm).real
