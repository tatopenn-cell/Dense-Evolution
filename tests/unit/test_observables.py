"""
Unit tests for dense_evolution/observables.py -- pauli_expectation and
pauli_sum_expectation, cross-checked against brute-force dense Pauli
matrices (kron products), not just against their own derivation.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator
from dense_evolution.observables import pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix

_PAULI_MATS = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _brute_force(psi, pauli_string):
    """Independent reference: build the full 2**n Pauli matrix via kron
    and compute <psi|P|psi> directly -- pauli_string[0] is the most
    significant factor, matching this package's qubit-0-is-MSB layout."""
    op = _PAULI_MATS[pauli_string[0]]
    for c in pauli_string[1:]:
        op = np.kron(op, _PAULI_MATS[c])
    return float(np.real(np.conj(psi) @ op @ psi))


class TestPauliExpectation:

    def test_matches_brute_force_dense_matrices(self):
        rng = np.random.default_rng(42)
        n_qubits, dim = 4, 16
        letters = ['I', 'X', 'Y', 'Z']
        for _ in range(200):
            psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            psi /= np.linalg.norm(psi)
            pauli_string = ''.join(rng.choice(letters) for _ in range(n_qubits))
            assert pauli_expectation(psi, pauli_string) == pytest.approx(
                _brute_force(psi, pauli_string), abs=1e-9)

    def test_identity_string_returns_norm(self):
        rng = np.random.default_rng(1)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        assert pauli_expectation(psi, 'III') == pytest.approx(1.0, abs=1e-9)

    def test_dict_form_matches_string_form(self):
        rng = np.random.default_rng(2)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        assert pauli_expectation(psi, {2: 'Z'}) == pytest.approx(
            pauli_expectation(psi, 'IIZI'), abs=1e-9)

    def test_pair_iterable_form_matches_string_form(self):
        rng = np.random.default_rng(3)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        assert pauli_expectation(psi, [(0, 'X'), (3, 'Z')]) == pytest.approx(
            pauli_expectation(psi, 'XIIZ'), abs=1e-9)

    def test_bell_state_zz_is_perfectly_correlated(self):
        sim = DenseSVSimulator(2, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        assert pauli_expectation(sim.get_statevector(), 'ZZ') == pytest.approx(1.0, abs=1e-9)

    def test_bell_state_xx_is_perfectly_correlated(self):
        sim = DenseSVSimulator(2, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        assert pauli_expectation(sim.get_statevector(), 'XX') == pytest.approx(1.0, abs=1e-9)

    def test_ghz3_zzz_vanishes(self):
        # <GHZ3|ZZZ|GHZ3> = 0.5*(<000|ZZZ|000> + <111|ZZZ|111>) = 0.5*(1-1) = 0
        sim = DenseSVSimulator(3, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1), ('cx', 1, 2)])
        assert pauli_expectation(sim.get_statevector(), 'ZZZ') == pytest.approx(0.0, abs=1e-9)

    def test_string_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pauli_expectation(np.array([1.0, 0, 0, 0]), 'ZZZ', n_qubits=2)

    def test_unknown_pauli_letter_raises(self):
        with pytest.raises(ValueError):
            pauli_expectation(np.array([1.0, 0, 0, 0]), {0: 'Q'})

    def test_qubit_out_of_range_raises(self):
        with pytest.raises(ValueError):
            pauli_expectation(np.array([1.0, 0, 0, 0]), {5: 'Z'})


class TestPauliSumExpectation:

    def test_matches_manual_weighted_sum(self):
        rng = np.random.default_rng(7)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        terms = [(1.0, 'ZZII'), (0.5, {0: 'X'}), (-0.3, 'IYYI')]
        expected = sum(c * pauli_expectation(psi, p) for c, p in terms)
        assert pauli_sum_expectation(psi, terms) == pytest.approx(expected, abs=1e-9)

    def test_empty_terms_is_zero(self):
        psi = np.array([1.0, 0, 0, 0])
        assert pauli_sum_expectation(psi, []) == 0.0


class TestPauliHamiltonianToMatrix:

    def test_matches_brute_force_dense_matrices(self):
        rng = np.random.default_rng(11)
        n_qubits, dim = 4, 16
        letters = ['I', 'X', 'Y', 'Z']
        for _ in range(50):
            n_terms = rng.integers(1, 6)
            terms = []
            expected = np.zeros((dim, dim), dtype=complex)
            for _ in range(n_terms):
                coeff = float(rng.normal())
                pauli_string = ''.join(rng.choice(letters) for _ in range(n_qubits))
                terms.append((coeff, pauli_string))
                op = _PAULI_MATS[pauli_string[0]]
                for c in pauli_string[1:]:
                    op = np.kron(op, _PAULI_MATS[c])
                expected += coeff * op
            H = pauli_hamiltonian_to_matrix(terms, n_qubits=n_qubits)
            assert np.allclose(H, expected, atol=1e-9)

    def test_is_hermitian(self):
        terms = [(1.0, 'ZZ'), (0.5, {0: 'X'}), (-0.3, {1: 'Y'})]
        H = pauli_hamiltonian_to_matrix(terms, n_qubits=2)
        assert np.allclose(H, H.conj().T)

    def test_matches_pauli_sum_expectation_on_a_real_state(self):
        rng = np.random.default_rng(3)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        terms = [(1.0, 'ZZI'), (0.5, {0: 'X'}), (-0.7, {1: 'Y', 2: 'Z'})]
        H = pauli_hamiltonian_to_matrix(terms, n_qubits=3)
        via_matrix = float(np.real(np.conj(psi) @ H @ psi))
        via_direct = pauli_sum_expectation(psi, terms)
        assert via_matrix == pytest.approx(via_direct, abs=1e-9)

    def test_bell_state_zz_ground_truth(self):
        sim = DenseSVSimulator(2, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        psi = sim.get_statevector()
        H = pauli_hamiltonian_to_matrix([(1.0, 'ZZ')], n_qubits=2)
        assert float(np.real(np.conj(psi) @ H @ psi)) == pytest.approx(1.0, abs=1e-9)

    def test_qubit_out_of_range_raises(self):
        with pytest.raises(ValueError):
            pauli_hamiltonian_to_matrix([(1.0, {5: 'Z'})], n_qubits=2)

    def test_invalid_n_qubits_raises(self):
        with pytest.raises(ValueError):
            pauli_hamiltonian_to_matrix([(1.0, {0: 'Z'})], n_qubits=0)
