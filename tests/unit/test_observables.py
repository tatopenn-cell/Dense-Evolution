"""
Unit tests for dense_evolution/observables.py -- pauli_expectation and
pauli_sum_expectation, cross-checked against brute-force dense Pauli
matrices (kron products), not just against their own derivation.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator
from dense_evolution.observables import (
    pauli_expectation, pauli_sum_expectation, pauli_hamiltonian_to_matrix, pauli_sum_matvec,
)
from dense_evolution.physics.observables import multiply_pauli_terms

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


class TestPauliSumMatvec:
    """prog.txt Sezione 4.1 -- the matrix-free H @ vector primitive behind
    ground_state_energy_sparse's scipy.sparse.linalg.eigsh path. Must
    agree with pauli_hamiltonian_to_matrix(terms, n_qubits) @ vector to
    machine precision -- that's the whole point, it's the same H, just
    never materialized as a (2**n, 2**n) array."""

    def test_matches_dense_matrix_matvec(self):
        rng = np.random.default_rng(21)
        n_qubits, dim = 4, 16
        letters = ['I', 'X', 'Y', 'Z']
        for _ in range(50):
            n_terms = rng.integers(1, 6)
            terms = [
                (float(rng.normal()), ''.join(rng.choice(letters) for _ in range(n_qubits)))
                for _ in range(n_terms)
            ]
            v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            H = pauli_hamiltonian_to_matrix(terms, n_qubits=n_qubits)
            expected = H @ v
            actual = pauli_sum_matvec(v, terms, n_qubits=n_qubits)
            assert np.allclose(actual, expected, atol=1e-9)

    def test_not_required_to_be_normalized(self):
        # A linear map, not an expectation value -- must work on an
        # arbitrary (non-unit-norm) vector, e.g. an intermediate Lanczos
        # vector from eigsh, not just a physical statevector.
        terms = [(1.0, 'ZZ'), (0.5, {0: 'X'})]
        v = np.array([2.0, 0.0, 0.0, 0.0], dtype=complex)
        H = pauli_hamiltonian_to_matrix(terms, n_qubits=2)
        assert np.allclose(pauli_sum_matvec(v, terms, n_qubits=2), H @ v)

    def test_bell_state_zz_ground_truth(self):
        sim = DenseSVSimulator(2, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        psi = sim.get_statevector()
        Hpsi = pauli_sum_matvec(psi, [(1.0, 'ZZ')], n_qubits=2)
        assert float(np.real(np.conj(psi) @ Hpsi)) == pytest.approx(1.0, abs=1e-9)

    def test_vector_length_not_a_power_of_two_raises(self):
        with pytest.raises(ValueError):
            pauli_sum_matvec(np.zeros(5, dtype=complex), [(1.0, {0: 'Z'})])

    def test_qubit_out_of_range_raises(self):
        with pytest.raises(ValueError):
            pauli_sum_matvec(np.zeros(4, dtype=complex), [(1.0, {5: 'Z'})])


class TestMultiplyPauliTerms:
    """Cross-checked against real matrix multiplication, not just the
    symbolic phase-tracking algebra's own self-consistency."""

    @pytest.mark.parametrize("a,b,expected_phase,expected_pauli", [
        ('X', 'X', 1.0, None), ('Y', 'Y', 1.0, None), ('Z', 'Z', 1.0, None),
        ('X', 'Y', 1j, 'Z'), ('Y', 'X', -1j, 'Z'),
        ('Y', 'Z', 1j, 'X'), ('Z', 'Y', -1j, 'X'),
        ('Z', 'X', 1j, 'Y'), ('X', 'Z', -1j, 'Y'),
    ])
    def test_all_nine_same_qubit_products_match_real_matrices(self, a, b, expected_phase, expected_pauli):
        A = pauli_hamiltonian_to_matrix([(1.0, a)], n_qubits=1)
        B = pauli_hamiltonian_to_matrix([(1.0, b)], n_qubits=1)
        coeff, merged = multiply_pauli_terms([(1.0, a), (1.0, b)])
        assert coeff == pytest.approx(expected_phase)
        assert merged == ({0: expected_pauli} if expected_pauli else {})
        combined = coeff * pauli_hamiltonian_to_matrix([(1.0, merged)] if merged else [(1.0, {})], n_qubits=1)
        assert np.allclose(A @ B, combined)

    def test_disjoint_qubits_no_phase_and_coefficients_multiply(self):
        coeff, merged = multiply_pauli_terms([(2.0, 'X'), (3.0, {1: 'Z'})])
        assert coeff == pytest.approx(6.0)
        assert merged == {0: 'X', 1: 'Z'}

    def test_three_factor_product_matches_matrix_chain(self):
        rng = np.random.default_rng(7)
        letters = ['X', 'Y', 'Z']
        for _ in range(50):
            a, b, c = rng.choice(letters, size=3)
            A = pauli_hamiltonian_to_matrix([(1.0, a)], n_qubits=1)
            B = pauli_hamiltonian_to_matrix([(1.0, b)], n_qubits=1)
            C = pauli_hamiltonian_to_matrix([(1.0, c)], n_qubits=1)
            coeff, merged = multiply_pauli_terms([(1.0, a), (1.0, b), (1.0, c)])
            combined = coeff * pauli_hamiltonian_to_matrix([(1.0, merged)] if merged else [(1.0, {})], n_qubits=1)
            assert np.allclose(A @ B @ C, combined), f"{a}*{b}*{c} mismatch"

    def test_order_matters(self):
        # X*Y = iZ, Y*X = -iZ -- same qubits, different order, opposite sign.
        coeff_xy, _ = multiply_pauli_terms([(1.0, 'X'), (1.0, 'Y')])
        coeff_yx, _ = multiply_pauli_terms([(1.0, 'Y'), (1.0, 'X')])
        assert coeff_xy == pytest.approx(-coeff_yx)
