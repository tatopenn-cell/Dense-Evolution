"""
Unit tests for dense_evolution/entropy.py -- partial_trace,
von_neumann_entropy, mutual_information. Checked against known-exact
textbook cases (Bell pair, GHZ state, product state), not just internal
self-consistency.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator, partial_trace, von_neumann_entropy, mutual_information, central_charge


def test_backward_compat_shim_entropy_reexports_public_api():
    # dense_evolution.entropy is the Phase 2 backward-compat shim left at
    # the old top-level path -- nothing in this suite imports through it
    # directly (everything sources these three from the top-level
    # dense_evolution package instead, which now gets them from
    # dense_evolution.physics.entropy), so without this the shim's own
    # lines go uncovered and a broken shim would go undetected by CI.
    from dense_evolution.entropy import partial_trace as shim_pt, von_neumann_entropy as shim_vne, mutual_information as shim_mi, central_charge as shim_cc
    assert shim_pt is partial_trace
    assert shim_vne is von_neumann_entropy
    assert shim_mi is mutual_information
    assert shim_cc is central_charge


class TestCentralCharge:
    """Real numbers reproduced directly from Dense-Evolution-Discovery
    Experiment 36 (scripts/central_charge_calabrese_cardy.py), not
    fabricated here -- N=12 open-boundary critical TFIM ground state."""

    N = 12
    # S(L) for L=2..10 at the self-dual CFT point g=1.0 -- real values,
    # regenerated directly from ising_exact_verification.py's own ground
    # state (de.partial_trace + de.von_neumann_entropy) for this test.
    LS_CRITICAL = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    S_CRITICAL = [0.33131561, 0.36444644, 0.38333007, 0.39335363, 0.39651621,
                  0.39335363, 0.38333007, 0.36444644, 0.33131561]

    def test_recovers_known_ising_central_charge_near_theory(self):
        c, r2 = central_charge(self.LS_CRITICAL, self.S_CRITICAL, self.N)
        assert r2 > 0.99, f"fit quality too low: R^2={r2}"
        assert abs(c - 0.5) < 0.15, f"extracted c={c:.4f} too far from theory 0.5"

    def test_off_critical_curve_gives_low_r_squared(self):
        # Real S(L) at g=1.8 (deep in the gapped/paramagnetic phase, same
        # N=12 open TFIM chain) -- entropy saturates (area law) instead
        # of following CFT log-scaling, so the fit should visibly
        # degrade, not report a spuriously clean c.
        s_off_critical = [0.10627, 0.10756, 0.10781, 0.10787, 0.10788,
                           0.10787, 0.10781, 0.10756, 0.10627]
        c, r2 = central_charge(self.LS_CRITICAL, s_off_critical, self.N)
        assert r2 < 0.95

    def test_symmetric_curve_gives_finite_fit(self):
        # S(L) = S(N-L) symmetry (required for any pure-state bipartition)
        # must not break the linear regression.
        c, r2 = central_charge([2, 4, 6, 8, 10], [0.4, 0.5, 0.55, 0.5, 0.4], self.N)
        assert np.isfinite(c) and np.isfinite(r2)


def _bell_state():
    sim = DenseSVSimulator(2, use_float32=False)
    sim.run_circuit([('h', 0), ('cx', 0, 1)])
    return sim.get_statevector()


def _ghz_state(n_qubits):
    sim = DenseSVSimulator(n_qubits, use_float32=False)
    ops = [('h', 0)] + [('cx', i, i + 1) for i in range(n_qubits - 1)]
    sim.run_circuit(ops)
    return sim.get_statevector()


def _product_state(n_qubits):
    """|+>|0>|0>... -- fully separable, zero entanglement anywhere."""
    sim = DenseSVSimulator(n_qubits, use_float32=False)
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
