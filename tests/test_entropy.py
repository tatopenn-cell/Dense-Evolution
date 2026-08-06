"""
Unit tests for dense_evolution/entropy.py -- partial_trace,
von_neumann_entropy, mutual_information. Checked against known-exact
textbook cases (Bell pair, GHZ state, product state), not just internal
self-consistency.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator, partial_trace, von_neumann_entropy, mutual_information


def _bell_state():
    sim = DenseSVSimulator(2, use_gpu=False, use_float32=False)
    sim.run_circuit([('h', 0), ('cx', 0, 1)])
    return sim.get_statevector()


def _ghz_state(n_qubits):
    sim = DenseSVSimulator(n_qubits, use_gpu=False, use_float32=False)
    ops = [('h', 0)] + [('cx', i, i + 1) for i in range(n_qubits - 1)]
    sim.run_circuit(ops)
    return sim.get_statevector()


def _product_state(n_qubits):
    """|+>|0>|0>... -- fully separable, zero entanglement anywhere."""
    sim = DenseSVSimulator(n_qubits, use_gpu=False, use_float32=False)
    sim.run_circuit([('h', 0)])
    return sim.get_statevector()


class TestPartialTrace:

    def test_full_system_trace_matches_pure_density_matrix(self):
        psi = _bell_state()
        rho = partial_trace(psi, n_qubits=2, keep_qubits=[0, 1])
        expected = np.outer(psi, psi.conj())
        assert np.max(np.abs(rho - expected)) == pytest.approx(0.0, abs=1e-10)

    def test_reduced_density_matrix_has_trace_one(self):
        psi = _bell_state()
        rho = partial_trace(psi, n_qubits=2, keep_qubits=[0])
        assert np.trace(rho).real == pytest.approx(1.0, abs=1e-10)

    def test_bell_pair_single_qubit_marginal_is_maximally_mixed(self):
        psi = _bell_state()
        rho = partial_trace(psi, n_qubits=2, keep_qubits=[0])
        assert np.max(np.abs(rho - np.eye(2) / 2)) == pytest.approx(0.0, abs=1e-10)


class TestVonNeumannEntropy:

    def test_pure_state_has_zero_entropy(self):
        psi = _bell_state()
        rho = partial_trace(psi, n_qubits=2, keep_qubits=[0, 1])
        assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-8)

    def test_maximally_mixed_qubit_has_ln2_entropy(self):
        psi = _bell_state()
        rho = partial_trace(psi, n_qubits=2, keep_qubits=[0])
        assert von_neumann_entropy(rho) == pytest.approx(np.log(2), abs=1e-8)


class TestMutualInformation:

    def test_bell_pair_gives_textbook_value_two_ln2(self):
        psi = _bell_state()
        mi = mutual_information(psi, n_qubits=2, qubits_a=[0], qubits_b=[1])
        assert mi == pytest.approx(2 * np.log(2), abs=1e-8)

    def test_ghz_state_two_of_three_qubits_share_ln2_mutual_information(self):
        """GHZ = (|000>+|111>)/sqrt(2): tracing out qubit 1 leaves qubits
        {0,2} in the *classically* correlated mixture 0.5|00><00|+0.5|11><11|
        (entropy ln2, not 0 -- GHZ correlations beyond two parties aren't
        fully captured pairwise), and each single qubit's marginal is also
        maximally mixed (entropy ln2) -- so I = ln2 + ln2 - ln2 = ln2, not
        the Bell pair's 2*ln2 (hand-verified, not assumed)."""
        psi = _ghz_state(3)
        mi = mutual_information(psi, n_qubits=3, qubits_a=[0], qubits_b=[2])
        assert mi == pytest.approx(np.log(2), abs=1e-8)

    def test_product_state_has_zero_mutual_information(self):
        psi = _product_state(3)
        mi = mutual_information(psi, n_qubits=3, qubits_a=[0], qubits_b=[1])
        assert mi == pytest.approx(0.0, abs=1e-8)

    def test_mutual_information_is_symmetric(self):
        psi = _ghz_state(3)
        mi_ab = mutual_information(psi, n_qubits=3, qubits_a=[0], qubits_b=[2])
        mi_ba = mutual_information(psi, n_qubits=3, qubits_a=[2], qubits_b=[0])
        assert mi_ab == pytest.approx(mi_ba, abs=1e-10)
