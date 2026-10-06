"""A differentiable-w.r.t.-nuclear-positions RHF total energy, composed
from basis.py + assembly.py + scf.py.

Two separate, deliberate pieces make this possible, neither of which is
a property of any one of those modules alone:

1. Schwarz screening (assembly.py's quartet_screening_indices) is a
   discrete, structural decision -- which shell quartets exist in the
   ERI sum at all -- and can't be part of a jax.grad trace (Python
   control flow can't run on a traced value). It's decided ONCE here,
   from the reference geometry passed to build_energy_fn, and reused as
   a fixed structure for every geometry the returned function is later
   called at.

2. scf.py's iterative convergence search (jax.lax.while_loop) can't be
   differentiated in reverse mode either -- scf_electronic_energy
   sidesteps that with the analytic Hartree-Fock gradient (Pople,
   Krishnan, Schlegel & Binkley, 1979) instead of backpropagating
   through the loop.

The returned function is only valid near the reference geometry: if the
nuclei move far enough that a previously-negligible shell-pair
interaction becomes non-negligible (or vice versa), the frozen
screening structure is stale and build_energy_fn should be called again
at the new geometry.
"""

import numpy as np

from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.assembly import (
    build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor, quartet_screening_indices,
)
from dense_evolution.native_hf.scf import (
    cuhf_electronic_energy, nuclear_repulsion_energy, scf_electronic_energy, uhf_electronic_energy,
)


def build_energy_fn(
    atomic_numbers: list[int], nuclear_charges: list[float], n_electrons: int,
    basis_name: str, reference_geometry_bohr: np.ndarray, screening_tol: float = 1e-12,
    method: str = "rhf", n_unpaired: int | None = None,
):
    """`method` is "rhf" (default), "uhf" or "cuhf"; for the open-shell methods
    `n_unpaired` defaults to `n_electrons % 2`. The UHF/CUHF gradients use the
    spin-resolved energy-weighted density (see scf.py)."""
    if method == "rhf":
        electronic = lambda S, H, V: scf_electronic_energy(S, H, V, n_electrons)
    elif method in ("uhf", "cuhf"):
        if n_unpaired is None:
            n_unpaired = n_electrons % 2
        if n_electrons < n_unpaired or (n_electrons - n_unpaired) % 2 != 0:
            raise ValueError(f"Invalid (n_electrons={n_electrons}, n_unpaired={n_unpaired})")
        n_alpha, n_beta = (n_electrons + n_unpaired) // 2, (n_electrons - n_unpaired) // 2
        fn = uhf_electronic_energy if method == "uhf" else cuhf_electronic_energy
        electronic = lambda S, H, V: fn(S, H, V, n_alpha, n_beta)
    else:
        raise ValueError(f"method must be 'rhf', 'uhf' or 'cuhf', got {method!r}")

    reference_shells = build_molecule_shells(atomic_numbers, np.asarray(reference_geometry_bohr), basis_name)
    quartet_indices = quartet_screening_indices(reference_shells, screening_tol)

    def energy_fn(geometry_bohr):
        shells = build_molecule_shells(atomic_numbers, geometry_bohr, basis_name)
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, nuclear_charges, geometry_bohr)
        repulsion = build_repulsion_tensor(shells, quartet_indices=quartet_indices)
        electronic_energy = electronic(S, H_core, repulsion)
        e_nuc = nuclear_repulsion_energy(nuclear_charges, geometry_bohr)
        return electronic_energy + e_nuc

    return energy_fn
