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
from dense_evolution.native_hf.scf import scf_electronic_energy, nuclear_repulsion_energy


def build_energy_fn(
    atomic_numbers: list[int], nuclear_charges: list[float], n_electrons: int,
    basis_name: str, reference_geometry_bohr: np.ndarray, screening_tol: float = 1e-12,
):
    reference_shells = build_molecule_shells(atomic_numbers, np.asarray(reference_geometry_bohr), basis_name)
    quartet_indices = quartet_screening_indices(reference_shells, screening_tol)

    def energy_fn(geometry_bohr):
        shells = build_molecule_shells(atomic_numbers, geometry_bohr, basis_name)
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, nuclear_charges, geometry_bohr)
        repulsion = build_repulsion_tensor(shells, quartet_indices=quartet_indices)
        electronic_energy = scf_electronic_energy(S, H_core, repulsion, n_electrons)
        e_nuc = nuclear_repulsion_energy(nuclear_charges, geometry_bohr)
        return electronic_energy + e_nuc

    return energy_fn
