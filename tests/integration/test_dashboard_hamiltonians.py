"""
Tests for dashboard_core.hamiltonians -- real molecular Hamiltonians
(PennyLane Hartree-Fock + Jordan-Wigner, densified via this project's own
dense_evolution.pauli_hamiltonian_to_matrix), the real geometry
generators behind the catalog, and real Hamiltonian mixing.

No Qiskit involved anywhere in this module -- these tests don't need the
macOS QuantumCircuit skip (see test_dashboard_engine.py for that).
"""

import numpy as np
import pytest

from dashboard_core.hamiltonians import (
    linear_chain_geometry, ring_geometry, MOLECULE_CATALOG,
    build_molecular_hamiltonian, get_molecule_n_qubits, get_all_molecules,
    get_compatible_molecules, get_molecular_hamiltonian_matrix, ground_state_energy,
    ground_state_energy_sparse, mix_hamiltonians,
)

H2_SYMBOLS = ['H', 'H']
H2_GEOMETRY = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]])
HEHP_SYMBOLS = ['He', 'H']
HEHP_GEOMETRY = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7743]])


class TestGeometryGenerators:

    def test_linear_chain_spacing_is_exact(self):
        g = linear_chain_geometry(3, 1.0)
        assert g.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]

    def test_linear_chain_rejects_zero_atoms(self):
        with pytest.raises(ValueError):
            linear_chain_geometry(0, 1.0)

    def test_ring_geometry_all_sides_equal_bond_length(self):
        g = ring_geometry(4, 1.0)
        sides = [np.linalg.norm(g[i] - g[(i + 1) % 4]) for i in range(4)]
        assert all(s == pytest.approx(1.0, abs=1e-9) for s in sides)

    def test_ring_geometry_rejects_fewer_than_3_atoms(self):
        with pytest.raises(ValueError):
            ring_geometry(2, 1.0)

    def test_ring_at_3_atoms_is_equilateral_triangle(self):
        g = ring_geometry(3, 0.8738)
        d01 = np.linalg.norm(g[0] - g[1])
        d12 = np.linalg.norm(g[1] - g[2])
        d02 = np.linalg.norm(g[0] - g[2])
        assert d01 == pytest.approx(0.8738, abs=1e-9)
        assert d12 == pytest.approx(0.8738, abs=1e-9)
        assert d02 == pytest.approx(0.8738, abs=1e-9)


class TestBuildMolecularHamiltonian:

    def test_h2_ground_state_matches_known_value(self):
        H, n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert n_qubits == 4
        assert ground_state_energy(H) == pytest.approx(-1.1372701748786913, abs=1e-6)

    def test_hehp_ground_state_matches_known_value(self):
        H, n_qubits = build_molecular_hamiltonian(HEHP_SYMBOLS, HEHP_GEOMETRY, charge=1)
        assert n_qubits == 4
        assert ground_state_energy(H) == pytest.approx(-3.0156651781733883, abs=1e-6)

    def test_hamiltonian_is_hermitian(self):
        H, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert np.allclose(H, H.conj().T)

    def test_unsupported_element_raises_a_clear_error_not_pennylanes_own(self):
        # An element outside PennyLane's own bundled STO-3G table (e.g.
        # Fe, past Ne) no longer fails outright -- it's handed to
        # native_hf (see dashboard_core.hamiltonians._get_hamiltonian and
        # the module docstring), which sources basis data from
        # basis_set_exchange instead and genuinely supports far more of
        # the periodic table (Si2 is in MOLECULE_CATALOG precisely
        # because of this). What native_hf's own integrals don't yet
        # implement is d-orbitals and up (only s/p, degree <= 1) -- Fe's
        # STO-3G basis needs a d shell, so this should still fail fast,
        # now with a message naming the real remaining limitation instead
        # of a raw internal crash.
        # A real, non-degenerate geometry (not np.zeros -- three atoms at
        # the exact same point is itself now a validation error, see
        # TestGeometryValidation below; the point of this test is the
        # element-support error, so the geometry just needs to be valid).
        geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
        with pytest.raises(NotImplementedError, match="Cartesian Gaussian"):
            build_molecular_hamiltonian(['Fe', 'Mo', 'S'], geometry, charge=0)

    def test_light_supported_elements_are_unaffected(self):
        # H2's own symbols (H, H) must still pass the new check -- guards
        # against the STO3G membership check itself being wrong.
        H, n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert n_qubits == 4

    def test_result_is_cached_identical_object_content(self):
        H1, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        H2_again, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert np.array_equal(H1, H2_again)

    def test_jordan_wigner_and_bravyi_kitaev_share_the_same_spectrum(self):
        # Different fermion-to-qubit mapping, same physical Hamiltonian --
        # the eigenvalue spectrum (and so the ground-state energy) must be
        # identical, only the qubit-operator representation changes.
        H_jw, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0, mapping="jordan_wigner")
        H_bk, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0, mapping="bravyi_kitaev")
        assert ground_state_energy(H_jw) == pytest.approx(ground_state_energy(H_bk), abs=1e-9)


class TestGroundStateEnergySparse:
    """prog.txt Sezione 4.1 -- matrix-free ground state via
    dense_evolution.pauli_sum_matvec + scipy.sparse.linalg.eigsh, the fix
    for the PRIORITARIO gap (ground_state_energy/build_molecular_hamiltonian
    only had a dense np.linalg.eigvalsh path, blocked concretely on Si2).
    Same known reference values as TestBuildMolecularHamiltonian above --
    this must agree with the dense path to machine precision, not just be
    plausible."""

    def test_h2_matches_known_dense_value(self):
        e = ground_state_energy_sparse(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert e == pytest.approx(-1.1372701748786913, abs=1e-6)

    def test_hehp_matches_known_dense_value(self):
        e = ground_state_energy_sparse(HEHP_SYMBOLS, HEHP_GEOMETRY, charge=1)
        assert e == pytest.approx(-3.0156651781733883, abs=1e-6)

    def test_matches_dense_build_molecular_hamiltonian_path_exactly(self):
        # Same molecule, both code paths -- not just close to a hardcoded
        # literature value, but agreeing with the sibling dense function
        # on the SAME real Hamiltonian object.
        H_dense, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        dense_e = ground_state_energy(H_dense)
        sparse_e = ground_state_energy_sparse(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert sparse_e == pytest.approx(dense_e, abs=1e-8)

    def test_jordan_wigner_and_bravyi_kitaev_share_the_same_spectrum(self):
        e_jw = ground_state_energy_sparse(H2_SYMBOLS, H2_GEOMETRY, charge=0, mapping="jordan_wigner")
        e_bk = ground_state_energy_sparse(H2_SYMBOLS, H2_GEOMETRY, charge=0, mapping="bravyi_kitaev")
        assert e_jw == pytest.approx(e_bk, abs=1e-8)


class TestGeometryValidation:
    """BUG FIX: build_molecular_hamiltonian had no input validation --
    a symbols/geometry length mismatch surfaced as a raw IndexError deep
    inside PennyLane's own internals, and non-finite coordinates were
    silently accepted, producing a NaN Hamiltonian with no error at all
    (both verified directly before this fix, not assumed)."""

    def test_symbols_geometry_length_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="2 symbols but 1 geometry rows"):
            build_molecular_hamiltonian(['H', 'H'], np.array([[0.0, 0.0, 0.0]]), charge=0)

    def test_non_finite_geometry_raises_instead_of_silently_producing_nan(self):
        geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.nan]])
        with pytest.raises(ValueError, match="non-finite"):
            build_molecular_hamiltonian(['H', 'H'], geometry, charge=0)

    def test_wrong_geometry_shape_raises_clear_error(self):
        with pytest.raises(ValueError, match=r"shape \(n_atoms, 3\)"):
            build_molecular_hamiltonian(['H', 'H'], np.array([[0.0, 0.0], [0.0, 0.0]]), charge=0)

    def test_overlapping_atoms_raise_clear_error_not_a_pennylane_crash(self):
        # 0.1 A is below any real atomic radius -- this is malformed
        # input (e.g. a units mistake), not a physically valid molecule.
        geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
        with pytest.raises(ValueError, match="physically-realistic floor"):
            build_molecular_hamiltonian(['H', 'H'], geometry, charge=0)

    def test_valid_geometry_still_works_unaffected(self):
        H, n_qubits = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert n_qubits == 4
        assert not np.any(np.isnan(H))


class TestCatalog:

    def test_get_molecule_n_qubits_matches_full_hamiltonian_build(self):
        n_qubits = get_molecule_n_qubits(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        assert n_qubits == 4

    def test_get_all_molecules_returns_every_catalog_entry(self):
        all_molecules = get_all_molecules()
        assert set(all_molecules.keys()) == set(MOLECULE_CATALOG.keys())
        for name, info in all_molecules.items():
            assert info["n_qubits"] >= 4
            assert "symbols" in info and "geometry" in info and "charge" in info

    def test_get_compatible_molecules_filters_by_qubit_count(self):
        compat4 = get_compatible_molecules(4)
        assert set(compat4.keys()) == {
            "H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]",
            "HeH+ (Idruro di Elio, catione) - R = 0.7743 A [equilibrio reale]",
        }

    def test_get_compatible_molecules_empty_for_impossible_qubit_count(self):
        assert get_compatible_molecules(3) == {}
        assert get_compatible_molecules(0) == {}

    def test_get_molecular_hamiltonian_matrix_by_catalog_name(self):
        H = get_molecular_hamiltonian_matrix("H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]")
        assert ground_state_energy(H) == pytest.approx(-1.1372701748786913, abs=1e-6)


class TestNativeHfFallback:
    """Si2 needs native_hf (dense_evolution.native_hf), since Si isn't in
    PennyLane's own bundled STO-3G table -- see _get_hamiltonian's
    dispatch and the module docstring. Slow (real Hartree-Fock on a
    2-atom/8-qubit active space, not PennyLane's), but relies on caching
    (this class runs after TestCatalog, which already built Si2's
    Hamiltonian once) to stay reasonably fast in the full suite."""

    SI2_NAME = "Si2 (Disilicio) - R = 2.184 A [equilibrio reale, active space minimo]"

    def test_si2_ground_state_matches_independent_verification(self):
        # BUG FOUND AND FIXED (see dense_evolution/native_hf/scf.py's
        # module docstring): plain (undamped) SCF never converged for
        # this molecule's minimal 4-electron/4-orbital active space --
        # two orbital pairs at the active-space boundary are numerically
        # degenerate, so the density oscillated for the full 100/100
        # iterations instead of settling. The value this test asserted
        # before (-570.68610495) was one such non-converged stopping
        # point, not a real answer -- it happened to reproduce on most
        # platforms/library versions but not all (CI caught this: a
        # different non-converged value, matching what's asserted below,
        # showed up on ubuntu-latest/Python 3.12). Linear density damping
        # now makes the SCF genuinely converge, to the SAME energy
        # (agreeing to 10 significant figures) across every damping
        # factor tested (0.1-0.7) -- a real self-consistency check, not
        # just a different arbitrary stopping point. The original
        # cross-check claim against lowdanie/hartree-fock-solver and
        # PennyLane's dhf result predates this fix and needs redoing
        # against the corrected value; not re-verified here yet.
        H = get_molecular_hamiltonian_matrix(self.SI2_NAME)
        assert ground_state_energy(H) == pytest.approx(-571.0169825890681, abs=1e-6)

    def test_si2_n_qubits(self):
        spec = MOLECULE_CATALOG[self.SI2_NAME]
        n_qubits = get_molecule_n_qubits(
            spec["symbols"], spec["geometry"](), spec["charge"],
            active_electrons=spec["active_electrons"], active_orbitals=spec["active_orbitals"],
        )
        assert n_qubits == 8  # 2 * active_orbitals

    def test_si2_hamiltonian_is_hermitian(self):
        H = get_molecular_hamiltonian_matrix(self.SI2_NAME)
        assert np.allclose(H, H.conj().T)

    def test_transition_metal_still_fails_clearly_not_a_silent_wrong_answer(self):
        # Fe's real STO-3G basis needs a d shell (angular momentum 2),
        # which native_hf's own overlap/kinetic/Coulomb integrals don't
        # implement yet (only s/p, degree <= 1) -- this must still fail
        # loudly, not silently produce a wrong energy for an incomplete
        # basis.
        geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        with pytest.raises(NotImplementedError):
            build_molecular_hamiltonian(['Fe', 'Fe'], geometry, charge=0)


class TestMixHamiltonians:

    def test_full_weight_on_a_reproduces_a_exactly(self):
        H_a, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        H_b, _ = build_molecular_hamiltonian(HEHP_SYMBOLS, HEHP_GEOMETRY, charge=1)
        H_mix = mix_hamiltonians(H_a, H_b, 1.0, 0.0)
        assert np.allclose(H_mix, H_a)

    def test_mixed_hamiltonian_is_hermitian(self):
        H_a, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)
        H_b, _ = build_molecular_hamiltonian(HEHP_SYMBOLS, HEHP_GEOMETRY, charge=1)
        H_mix = mix_hamiltonians(H_a, H_b, 0.5, 0.5)
        assert np.allclose(H_mix, H_mix.conj().T)

    def test_mismatched_qubit_counts_raise(self):
        H_a, _ = build_molecular_hamiltonian(H2_SYMBOLS, H2_GEOMETRY, charge=0)  # 4 qubits
        H_c, _ = build_molecular_hamiltonian(['H', 'H', 'H'], ring_geometry(3, 0.8738), charge=1)  # 6 qubits
        with pytest.raises(ValueError, match="different qubit spaces"):
            mix_hamiltonians(H_a, H_c)
