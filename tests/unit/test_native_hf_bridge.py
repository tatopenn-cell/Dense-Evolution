"""
Unit tests for dense_evolution/native_hf/bridge.py's AO->MO integral
transformation (_ao_to_mo) -- flagged by an external review as the part
of this module with the weakest test coverage: a delicate 4-index
einsum + swapaxes where an index-ordering mistake could produce a
plausible-looking but numerically wrong Hamiltonian. Cross-checked
against two independent references, not just re-run against itself:
a plain sequential one-index-at-a-time transformation (a genuinely
different algorithm computing the same quantity), and a real physical
invariant (the transform must not change the total electronic energy).
"""
import numpy as np
import pytest

from dense_evolution.native_hf.bridge import _ao_to_mo, _apply_active_space
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.assembly import (
    build_overlap_matrix, build_core_hamiltonian, build_repulsion_tensor,
)
from dense_evolution.native_hf.scf import run_scf

_BOHR_PER_ANGSTROM = 1.8897259886


def _sequential_ao_to_mo(repulsion: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Independent reference for the two-electron AO->MO transform: one
    index contracted at a time (the standard textbook N^5 sequential
    transform), instead of _ao_to_mo's single combined 5-tensor einsum
    -- a genuinely different sequence of operations computing the same
    quantity, not a copy of the code under test."""
    step1 = np.einsum('ab,bdeg->adeg', C.T, repulsion)
    step2 = np.einsum('cd,adeg->aceg', C.T, step1)
    step3 = np.einsum('ef,aceg->acfg', C, step2)
    step4 = np.einsum('gh,acfg->acfh', C, step3)
    return step4


class TestAoToMo:

    def test_one_electron_matches_plain_matrix_product(self):
        rng = np.random.default_rng(0)
        n = 5
        H_core = rng.standard_normal((n, n))
        H_core = H_core + H_core.T
        repulsion = rng.standard_normal((n, n, n, n))
        C = rng.standard_normal((n, n))

        one, _ = _ao_to_mo(H_core, repulsion, C)
        assert np.allclose(one, C.T @ H_core @ C, atol=1e-12)

    def test_two_electron_matches_independent_sequential_transform(self):
        rng = np.random.default_rng(1)
        n = 4
        # A real ERI tensor has 8-fold permutational symmetry -- build one
        # that does, rather than an arbitrary 4-index array, so this test
        # exercises the same kind of input _ao_to_mo actually sees.
        repulsion = rng.standard_normal((n, n, n, n))
        repulsion = repulsion + repulsion.transpose(1, 0, 2, 3)
        repulsion = repulsion + repulsion.transpose(0, 1, 3, 2)
        repulsion = repulsion + repulsion.transpose(2, 3, 0, 1)
        C = rng.standard_normal((n, n))
        H_core = np.zeros((n, n))

        _, two = _ao_to_mo(H_core, repulsion, C)
        expected = _sequential_ao_to_mo(repulsion, C)
        # _ao_to_mo's own swapaxes(1, 3) undone here to compare against
        # the un-swapped sequential reference directly.
        assert np.allclose(np.swapaxes(two, 1, 3), expected, atol=1e-10)

    def test_full_space_transform_preserves_real_h2_electronic_energy(self):
        # The real physical invariant a broken index order would violate:
        # transforming AO integrals to the MO basis, then reducing to the
        # SAME full space (no core frozen, every orbital active), must
        # reproduce the identical total electronic energy the AO-basis
        # SCF already converged to -- a basis change alone cannot alter a
        # physical observable. Uses the real H2/STO-3G integrals build_qubit_hamiltonian
        # itself would use, not a synthetic system.
        geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]]) * _BOHR_PER_ANGSTROM
        shells = build_molecule_shells([1, 1], geometry_bohr, "sto-3g")
        S = build_overlap_matrix(shells)
        H_core = build_core_hamiltonian(shells, [1.0, 1.0], geometry_bohr)
        repulsion = build_repulsion_tensor(shells)
        hf_result = run_scf(S, H_core, repulsion, 2, [1.0, 1.0], geometry_bohr)

        one, two = _ao_to_mo(H_core, repulsion, hf_result.orbital_coefficients)
        n_orbitals = one.shape[0]
        core_constant, one_active, two_active = _apply_active_space(
            0.0, one, two, [], list(range(n_orbitals))
        )

        n_occupied_pairs = 1  # H2: 2 electrons
        E_mo = core_constant
        for i in range(n_occupied_pairs):
            E_mo += 2 * one_active[i, i]
            for j in range(n_occupied_pairs):
                E_mo += 2 * two_active[i, j, j, i] - two_active[i, j, i, j]

        assert E_mo == pytest.approx(hf_result.electronic_energy, abs=1e-10)
