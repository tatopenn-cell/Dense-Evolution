"""
Unit tests for dense_evolution/fermions.py -- Majorana-fermion -> qubit
(Jordan-Wigner) mapping. Cross-checked against the actual dense Pauli
matrices via pauli_hamiltonian_to_matrix, not just the textbook formula.
"""
import numpy as np
import pytest

from dense_evolution import majorana_pauli_terms
from dense_evolution.observables import pauli_hamiltonian_to_matrix


def _chi_matrix(mode_index, n_qubits):
    coeff, pauli_dict = majorana_pauli_terms(mode_index, n_qubits)
    return coeff * pauli_hamiltonian_to_matrix([(1.0, pauli_dict)], n_qubits)


class TestMajoranaPauliTerms:

    def test_anticommutation_relation_exact(self):
        """{chi_a, chi_b} = 2*delta_ab*I for every pair, N=6 -> 3 qubits."""
        n_qubits = 3
        n_majorana = 2 * n_qubits
        chis = [_chi_matrix(m, n_qubits) for m in range(1, n_majorana + 1)]
        identity = np.eye(2 ** n_qubits, dtype=complex)
        max_error = 0.0
        for a in range(n_majorana):
            for b in range(n_majorana):
                anticomm = chis[a] @ chis[b] + chis[b] @ chis[a]
                expected = 2 * identity if a == b else np.zeros_like(identity)
                max_error = max(max_error, float(np.max(np.abs(anticomm - expected))))
        assert max_error == pytest.approx(0.0, abs=1e-10)

    def test_each_mode_is_hermitian(self):
        n_qubits = 3
        for m in range(1, 2 * n_qubits + 1):
            chi = _chi_matrix(m, n_qubits)
            assert np.max(np.abs(chi - chi.conj().T)) == pytest.approx(0.0, abs=1e-12)

    def test_each_mode_squares_to_identity(self):
        n_qubits = 3
        identity = np.eye(2 ** n_qubits, dtype=complex)
        for m in range(1, 2 * n_qubits + 1):
            chi = _chi_matrix(m, n_qubits)
            assert np.max(np.abs(chi @ chi - identity)) == pytest.approx(0.0, abs=1e-12)

    def test_odd_mode_is_x_on_its_qubit(self):
        # chi_1 = X_0 (j=0, is_even=False)
        coeff, pauli_dict = majorana_pauli_terms(1, n_qubits=3)
        assert coeff == 1.0
        assert pauli_dict == {0: 'X'}

    def test_even_mode_is_y_on_its_qubit_with_z_string(self):
        # chi_4: mode_index=4 -> j=(4-1)//2=1, is_even=True -> Z_0 Y_1
        coeff, pauli_dict = majorana_pauli_terms(4, n_qubits=3)
        assert coeff == 1.0
        assert pauli_dict == {0: 'Z', 1: 'Y'}

    def test_out_of_range_mode_index_raises(self):
        with pytest.raises(ValueError):
            majorana_pauli_terms(0, n_qubits=3)
        with pytest.raises(ValueError):
            majorana_pauli_terms(7, n_qubits=3)

    def test_boundary_mode_indices_are_valid(self):
        n_qubits = 3
        majorana_pauli_terms(1, n_qubits)
        majorana_pauli_terms(2 * n_qubits, n_qubits)
