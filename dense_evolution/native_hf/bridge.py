"""Bridge from our native Hartree-Fock result to a PennyLane qubit Hamiltonian.

The expensive part (Hartree-Fock: integrals + SCF, everything in this
package) is entirely ours. Second quantization and the Jordan-Wigner
mapping are cheap (profiled at under 2 seconds even for Si2 -- see
dense_evolution/native_hf/__init__.py's module docstring) and PennyLane
already does them well via public functions, so we call those directly
instead of reimplementing them.
"""

import numpy as np
import pennylane as qml
import pennylane.qchem.observable_hf as _pl_observable

from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor
from dense_evolution.native_hf.scf import run_scf

_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _ao_to_mo(H_core: np.ndarray, repulsion: np.ndarray, C: np.ndarray):
    one = np.einsum("qr,rs,st->qt", C.T, H_core, C)
    two = np.swapaxes(
        np.einsum("ab,cd,bdeg,ef,gh->acfh", C.T, C.T, repulsion, C, C),
        1, 3,
    )
    return one, two


def _apply_active_space(core_constant: float, one: np.ndarray, two: np.ndarray, core_idx, active_idx):
    for i in core_idx:
        core_constant = core_constant + 2 * one[i][i]
        for j in core_idx:
            core_constant = core_constant + 2 * two[i][j][j][i] - two[i][j][i][j]

    for p in active_idx:
        for q in active_idx:
            for i in core_idx:
                delta = np.zeros(one.shape)
                delta[p, q] = 1.0
                one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * delta

    one_active = one[np.ix_(active_idx, active_idx)]
    two_active = two[np.ix_(active_idx, active_idx, active_idx, active_idx)]
    return core_constant, one_active, two_active


def build_qubit_hamiltonian(
    atomic_numbers: list[int],
    geometry_angstrom: np.ndarray,
    n_electrons: int,
    active_electrons: int = None,
    active_orbitals: int = None,
    basis_name: str = "sto-3g",
    cutoff: float = 1e-12,
):
    """Runs native Hartree-Fock, then hands the result to PennyLane for
    second quantization + Jordan-Wigner mapping.

    Returns:
        (qubit_hamiltonian, n_qubits, hf_result) -- hf_result is the
        native_hf.scf.HFResult, useful for e.g. reporting the SCF energy
        alongside the post-mapping ground-state energy.
    """
    geometry_bohr = np.asarray(geometry_angstrom) * _BOHR_PER_ANGSTROM
    shells = build_molecule_shells(atomic_numbers, geometry_bohr, basis_name)

    S = build_overlap_matrix(shells)
    H_core = build_core_hamiltonian(shells, [float(z) for z in atomic_numbers], geometry_bohr)
    repulsion = build_repulsion_tensor(shells)

    hf_result = run_scf(S, H_core, repulsion, n_electrons, [float(z) for z in atomic_numbers], geometry_bohr)

    one, two = _ao_to_mo(H_core, repulsion, hf_result.orbital_coefficients)
    n_orbitals = one.shape[0]

    if active_electrons is None and active_orbitals is None:
        core_idx, active_idx = [], list(range(n_orbitals))
    else:
        core_idx, active_idx = qml.qchem.active_space(
            n_electrons, n_orbitals, active_electrons=active_electrons, active_orbitals=active_orbitals
        )

    core_constant, one_active, two_active = _apply_active_space(
        hf_result.nuclear_repulsion_energy, one, two, core_idx, active_idx
    )

    fermi_op = _pl_observable.fermionic_observable(
        np.array([core_constant]), one_active, two_active, cutoff
    )
    qubit_hamiltonian = qml.jordan_wigner(fermi_op)
    n_qubits = 2 * len(active_idx)

    return qubit_hamiltonian, n_qubits, hf_result
