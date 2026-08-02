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
    mix_hamiltonians,
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
        # PennyLane's own error for a missing STO-3G basis (e.g. Fe, Mo --
        # heavier than Ne) is real but written for someone already inside
        # its own codebase ("consider using load_data=True ..."). This
        # should fail fast with a message naming the actual unsupported
        # symbols and framing it as a basis-set gap, not a dense_evolution
        # limitation -- and before any Hartree-Fock/densification work.
        with pytest.raises(ValueError, match="No built-in STO-3G basis data for: Fe"):
            build_molecular_hamiltonian(['Fe', 'Mo', 'S'], np.zeros((3, 3)), charge=0)

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
