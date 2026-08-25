"""
Unit tests for dense_evolution/circuits/uccsd.py -- native (PennyLane-free)
UCCSD single/double excitation circuits.

Ground truth throughout is built independently: raw Jordan-Wigner ladder
operators (a_p, a_p^dagger) constructed directly with numpy Kronecker
products, NOT via majorana_pauli_terms and NOT via PennyLane -- so a bug
shared between the module under test and its own internal generator
derivation would still be caught here.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from dense_evolution import DenseSVSimulator
from dense_evolution.circuits.uccsd import find_excitations, single_excitation_ops, double_excitation_ops

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _a_dagger(p, n):
    zstring = _kron_all([Z] * p) if p > 0 else np.array([[1]], dtype=complex)
    op = np.kron(zstring, (X - 1j * Y) / 2)
    if n - p - 1 > 0:
        op = np.kron(op, np.eye(2 ** (n - p - 1), dtype=complex))
    return op


def _a(p, n):
    zstring = _kron_all([Z] * p) if p > 0 else np.array([[1]], dtype=complex)
    op = np.kron(zstring, (X + 1j * Y) / 2)
    if n - p - 1 > 0:
        op = np.kron(op, np.eye(2 ** (n - p - 1), dtype=complex))
    return op


def _single_generator_matrix(p, q, n):
    return _a_dagger(p, n) @ _a(q, n) - _a_dagger(q, n) @ _a(p, n)


def _double_generator_matrix(p, q, r, s, n):
    return (_a_dagger(p, n) @ _a_dagger(q, n) @ _a(r, n) @ _a(s, n)
            - _a_dagger(s, n) @ _a_dagger(r, n) @ _a(q, n) @ _a(p, n))


def _run_ops(ops, n, basis_index):
    sim = DenseSVSimulator(n_qubits=n)
    sim.sv = np.zeros(2 ** n, dtype=complex)
    sim.sv[basis_index] = 1.0
    sim.run_circuit(ops)
    return sim.sv


class TestFindExcitations:

    @pytest.mark.parametrize("electrons,n_qubits", [
        (2, 4), (2, 6), (4, 8), (2, 8), (4, 6),
    ])
    def test_matches_pennylane(self, electrons, n_qubits):
        pennylane = pytest.importorskip("pennylane")
        expected_singles, expected_doubles = pennylane.qchem.excitations(electrons, n_qubits)
        singles, doubles = find_excitations(electrons, n_qubits)
        assert singles == expected_singles
        assert doubles == expected_doubles


class TestSingleExcitation:

    @pytest.mark.parametrize("p,q,n", [
        (0, 1, 3), (0, 2, 3), (1, 4, 6), (0, 5, 6), (2, 3, 4),
    ])
    def test_exact_against_direct_ladder_operators(self, p, q, n):
        """Every basis state, several (p, q, n) including adjacent and
        long Jordan-Wigner Z-strings -- must match to floating-point
        precision, not just approximately."""
        theta = 0.4173
        G = _single_generator_matrix(p, q, n)
        target = expm(theta * G)
        ops = single_excitation_ops(p, q, theta)
        for basis in range(2 ** n):
            out = _run_ops(ops, n, basis)
            expected = target[:, basis]
            assert np.max(np.abs(out - expected)) < 1e-10

    def test_rejects_p_not_less_than_q(self):
        with pytest.raises(ValueError, match="p < q"):
            single_excitation_ops(2, 1, 0.5)

    def test_zero_theta_is_identity(self):
        ops = single_excitation_ops(0, 2, 0.0)
        out = _run_ops(ops, 3, 0)
        expected = np.zeros(8, dtype=complex); expected[0] = 1.0
        assert np.max(np.abs(out - expected)) < 1e-12


class TestDoubleExcitationClosedForm:

    @pytest.mark.parametrize("p,q,r,s,n_logical", [
        (0, 1, 2, 3, 4),   # fully adjacent (e.g. H2 minimal basis)
        (0, 1, 4, 5, 6),   # adjacent pairs, large HOMO-LUMO gap
        (2, 3, 4, 5, 6),
    ])
    def test_exact_against_direct_ladder_operators(self, p, q, r, s, n_logical):
        theta = 0.317
        a1, a2 = n_logical, n_logical + 1
        n = n_logical + 2
        G = _double_generator_matrix(p, q, r, s, n)
        target = expm(theta * G)
        ops = double_excitation_ops(p, q, r, s, theta, a1, a2)
        for basis in range(0, 2 ** n, 4):  # ancillas start at |00> only
            out = _run_ops(ops, n, basis)
            expected = target[:, basis]
            assert np.max(np.abs(out - expected)) < 1e-9

    def test_ancillas_return_to_zero(self):
        """Ancilla qubits must come back to |0> regardless of the
        logical register's state, so callers can reuse them."""
        n_logical, theta = 4, 0.6
        a1, a2 = 4, 5
        n = 6
        ops = double_excitation_ops(0, 1, 2, 3, theta, a1, a2)
        for basis in range(0, 2 ** n, 4):
            out = _run_ops(ops, n, basis)
            prob_ancilla_nonzero = np.sum(np.abs(out) ** 2) - sum(
                np.abs(out[i]) ** 2 for i in range(0, 2 ** n, 4)
            )
            assert prob_ancilla_nonzero < 1e-10


class TestDoubleExcitationPerTermFallback:

    @pytest.mark.parametrize("p,q,r,s,n", [
        (0, 2, 4, 6, 7),   # non-adjacent occupied pair
        (0, 1, 3, 5, 6),   # non-adjacent virtual pair
        (1, 3, 5, 7, 8),   # both pairs non-adjacent
        (0, 2, 3, 5, 6),
    ])
    def test_exact_against_direct_ladder_operators(self, p, q, r, s, n):
        """The fallback path (used whenever the occupied or virtual pair
        isn't adjacent) is verified exact here too, not just 'close' --
        see the module docstring for why this held up empirically."""
        theta = 0.9137
        G = _double_generator_matrix(p, q, r, s, n)
        target = expm(theta * G)
        ops = double_excitation_ops(p, q, r, s, theta, ancilla1=97, ancilla2=98)
        for basis in range(2 ** n):
            out = _run_ops(ops, n, basis)
            expected = target[:, basis]
            assert np.max(np.abs(out - expected)) < 1e-9

    def test_rejects_unordered_indices(self):
        with pytest.raises(ValueError, match=r"p<q<r<s"):
            double_excitation_ops(0, 2, 1, 3, 0.5, 10, 11)

    def test_no_ancillas_forces_per_term_path_even_when_adjacent(self):
        """Omitting ancilla1/ancilla2 must give the ancilla-free path
        unconditionally, even for an adjacent pair that would otherwise
        qualify for the closed form -- e.g. a caller with no spare
        qubits to offer."""
        p, q, r, s, n, theta = 0, 1, 2, 3, 4, 0.6
        G = _double_generator_matrix(p, q, r, s, n)
        target = expm(theta * G)
        ops = double_excitation_ops(p, q, r, s, theta)  # ancillas omitted
        assert all(name != 'ccx' for (name, *_) in ops)  # sanity: no Toffoli in this path
        for basis in range(2 ** n):
            out = _run_ops(ops, n, basis)
            assert np.max(np.abs(out - target[:, basis])) < 1e-9
