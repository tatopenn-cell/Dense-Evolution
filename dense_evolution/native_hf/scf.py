"""Restricted Hartree-Fock self-consistent field loop (Roothaan-Hall).

Standard textbook algorithm (e.g. Szabo & Ostlund, "Modern Quantum
Chemistry", ch. 3): orthogonalize the AO basis via S^(-1/2), build the
Fock matrix F = H_core + 2J - K from the current density, diagonalize
in the orthogonal basis, form a new density, repeat to convergence.
This part is genuinely simple compared to the integral evaluation and
doesn't need vectorizing -- a closed-shell molecule's SCF loop is a few
dozen matrix multiplies on an N x N matrix where N is a few tens at
most for STO-3G, nowhere near where PennyLane's implementation loses
its time (which is entirely in building H_core/repulsion tensor, done
once in assembly.py, not in this loop).
"""

import dataclasses

import numpy as np


@dataclasses.dataclass
class HFResult:
    converged: bool
    n_iterations: int
    electronic_energy: float
    nuclear_repulsion_energy: float
    total_energy: float
    orbital_energies: np.ndarray
    orbital_coefficients: np.ndarray  # C, shape (n_basis, n_basis)
    density_matrix: np.ndarray


def nuclear_repulsion_energy(nuclear_charges: list[float], nuclear_positions: np.ndarray) -> float:
    energy = 0.0
    n = len(nuclear_charges)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(nuclear_positions[i] - nuclear_positions[j])
            energy += nuclear_charges[i] * nuclear_charges[j] / r
    return energy


def _orthogonalizer(S: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(S)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def _density_from_coefficients(C: np.ndarray, n_occupied_pairs: int) -> np.ndarray:
    C_occ = C[:, :n_occupied_pairs]
    return C_occ @ C_occ.T


def run_scf(
    S: np.ndarray,
    H_core: np.ndarray,
    repulsion: np.ndarray,
    n_electrons: int,
    nuclear_charges: list[float],
    nuclear_positions: np.ndarray,
    max_iterations: int = 100,
    convergence_tol: float = 1e-10,
) -> HFResult:
    if n_electrons % 2 != 0:
        raise ValueError("Only closed-shell (even electron count) systems are supported.")
    n_occupied_pairs = n_electrons // 2

    X = _orthogonalizer(S)

    orbital_energies, C_ortho = np.linalg.eigh(X.T @ H_core @ X)
    C = X @ C_ortho
    P = _density_from_coefficients(C, n_occupied_pairs)

    converged = False
    for iteration in range(1, max_iterations + 1):
        J = np.einsum("pqrs,rs->pq", repulsion, P)
        K = np.einsum("prqs,rs->pq", repulsion, P)
        F = H_core + 2.0 * J - K

        orbital_energies, C_ortho = np.linalg.eigh(X.T @ F @ X)
        C = X @ C_ortho
        P_new = _density_from_coefficients(C, n_occupied_pairs)

        if np.linalg.norm(P_new - P) < convergence_tol:
            P = P_new
            converged = True
            break
        P = P_new

    electronic_energy = float(np.sum(P * (H_core + F)))
    e_nuc = nuclear_repulsion_energy(nuclear_charges, nuclear_positions)

    return HFResult(
        converged=converged,
        n_iterations=iteration,
        electronic_energy=electronic_energy,
        nuclear_repulsion_energy=e_nuc,
        total_energy=electronic_energy + e_nuc,
        orbital_energies=orbital_energies,
        orbital_coefficients=C,
        density_matrix=P,
    )
