"""
Unit tests for dense_evolution/trotter.py -- pauli_rotation_ops (exact
exp(-i*angle*P) as a gate circuit) and trotter_evolve_ops (first-order
Trotter product formula for exp(-i*H*t)). Checked against exact
scipy.linalg.expm, not just internal consistency.
"""
import numpy as np
import pytest
import scipy.linalg

from dense_evolution import DenseSVSimulator, pauli_rotation_ops, trotter_evolve_ops
from dense_evolution.measurement import statevector_fidelity
from dense_evolution.observables import pauli_hamiltonian_to_matrix


def _random_state(n_qubits, seed):
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


def _exact_rotation_matrix(pauli_dict, angle, n_qubits):
    """exp(-i*angle*P) as a dense matrix, independent reference via expm."""
    P = pauli_hamiltonian_to_matrix([(1.0, pauli_dict)], n_qubits)
    return scipy.linalg.expm(-1j * angle * P)


def _run_ops_on_state(psi, n_qubits, ops):
    sim = DenseSVSimulator(n_qubits, use_gpu=False, use_float32=False)
    sim.set_initial_state(psi)
    sim.run_circuit(ops)
    return sim.get_statevector()


class TestPauliRotationOps:

    @pytest.mark.parametrize("pauli_dict", [
        {0: 'X'},
        {0: 'Y'},
        {0: 'Z'},
        {0: 'X', 1: 'Y'},
        {0: 'Y', 1: 'X', 2: 'Z', 3: 'Y'},  # 4-qubit mixed string, SYK-like
    ])
    def test_fidelity_one_against_exact_expm(self, pauli_dict):
        n_qubits = max(pauli_dict.keys()) + 1
        angle = 0.37
        psi0 = _random_state(n_qubits, seed=hash(tuple(sorted(pauli_dict.items()))) % 2**31)

        ops = pauli_rotation_ops(pauli_dict, angle)
        psi_circuit = _run_ops_on_state(psi0, n_qubits, ops)

        exact_U = _exact_rotation_matrix(pauli_dict, angle, n_qubits)
        psi_exact = exact_U @ psi0

        fidelity = statevector_fidelity(psi_circuit, psi_exact)
        assert fidelity == pytest.approx(1.0, abs=1e-9)

    def test_empty_pauli_dict_returns_no_ops(self):
        assert pauli_rotation_ops({}, angle=1.23) == []

    def test_single_z_rotation_matches_rz_gate_convention(self):
        """This package's rz(theta) = exp(-i*theta/2*Z), so
        pauli_rotation_ops({0: 'Z'}, angle) must equal a bare rz(2*angle)."""
        n_qubits = 1
        angle = 0.9
        psi0 = _random_state(n_qubits, seed=7)
        ops = pauli_rotation_ops({0: 'Z'}, angle)
        assert ops == [('rz', 0, 2 * angle)]
        psi_circuit = _run_ops_on_state(psi0, n_qubits, ops)
        exact_U = _exact_rotation_matrix({0: 'Z'}, angle, n_qubits)
        assert statevector_fidelity(psi_circuit, exact_U @ psi0) == pytest.approx(1.0, abs=1e-9)


class TestTrotterEvolveOps:

    def test_converges_to_exact_evolution_as_steps_increase(self):
        """Infidelity should shrink monotonically (roughly quadratically)
        as n_steps grows, for a real multi-term, non-commuting Hamiltonian."""
        n_qubits = 3
        terms = [
            (0.6, {0: 'X', 1: 'Y'}),
            (-0.4, {1: 'Z', 2: 'X'}),
            (0.3, {0: 'Y', 2: 'Z'}),
        ]
        t = 0.5
        psi0 = _random_state(n_qubits, seed=11)

        H = sum(c * pauli_hamiltonian_to_matrix([(1.0, p)], n_qubits) for c, p in terms)
        psi_exact = scipy.linalg.expm(-1j * H * t) @ psi0

        infidelities = []
        for n_steps in (1, 2, 4, 8, 16):
            ops = trotter_evolve_ops(terms, t, n_steps)
            psi_trotter = _run_ops_on_state(psi0, n_qubits, ops)
            fidelity = statevector_fidelity(psi_trotter, psi_exact)
            infidelities.append(1.0 - fidelity)

        # monotonically non-increasing as step count doubles
        for earlier, later in zip(infidelities, infidelities[1:]):
            assert later <= earlier + 1e-12
        # converges close to exact by 16 steps
        assert infidelities[-1] < 1e-4

    def test_single_term_trotter_matches_pauli_rotation_ops_exactly(self):
        """With one term and one step, trotter_evolve_ops must reduce to
        exactly pauli_rotation_ops (no approximation to make)."""
        pauli_dict = {0: 'X', 1: 'Z'}
        terms = [(1.0, pauli_dict)]
        t, n_steps = 0.8, 1
        assert trotter_evolve_ops(terms, t, n_steps) == pauli_rotation_ops(pauli_dict, t)

    def test_gate_count_scales_with_terms_and_steps(self):
        terms = [(1.0, {0: 'X'}), (1.0, {0: 'Y', 1: 'Z'})]
        ops_per_step = len(pauli_rotation_ops({0: 'X'}, 1.0)) + len(pauli_rotation_ops({0: 'Y', 1: 'Z'}, 1.0))
        ops = trotter_evolve_ops(terms, t=1.0, n_steps=5)
        assert len(ops) == ops_per_step * 5

    def test_invalid_order_raises(self):
        terms = [(1.0, {0: 'X'})]
        with pytest.raises(ValueError):
            trotter_evolve_ops(terms, t=1.0, n_steps=1, order=3)


class TestTrotterEvolveOpsSecondOrder:

    def test_order_2_converges_faster_than_order_1(self):
        """Second-order (Strang) infidelity should shrink ~16x per step
        doubling (quartic in state overlap), clearly faster than
        order=1's ~4x (quadratic), for the same real, non-trivial,
        non-commuting Hamiltonian used in the order=1 convergence test."""
        n_qubits = 3
        terms = [
            (0.6, {0: 'X', 1: 'Y'}),
            (-0.4, {1: 'Z', 2: 'X'}),
            (0.3, {0: 'Y', 2: 'Z'}),
        ]
        t = 0.5
        psi0 = _random_state(n_qubits, seed=11)

        H = sum(c * pauli_hamiltonian_to_matrix([(1.0, p)], n_qubits) for c, p in terms)
        psi_exact = scipy.linalg.expm(-1j * H * t) @ psi0

        infidelities = []
        for n_steps in (1, 2, 4, 8):
            ops = trotter_evolve_ops(terms, t, n_steps, order=2)
            psi_trotter = _run_ops_on_state(psi0, n_qubits, ops)
            fidelity = statevector_fidelity(psi_trotter, psi_exact)
            infidelities.append(1.0 - fidelity)

        for earlier, later in zip(infidelities, infidelities[1:]):
            assert later <= earlier + 1e-12
        # quartic convergence: infidelity ratio between consecutive
        # doublings should be well below order=1's ~4x -- 8x is a safe
        # margin that still clearly distinguishes it from first order.
        for earlier, later in zip(infidelities[:-1], infidelities[1:]):
            if earlier > 1e-14:
                assert later < earlier / 8

    def test_order_2_more_accurate_than_order_1_at_matched_step_count(self):
        n_qubits = 3
        terms = [
            (0.6, {0: 'X', 1: 'Y'}),
            (-0.4, {1: 'Z', 2: 'X'}),
            (0.3, {0: 'Y', 2: 'Z'}),
        ]
        t = 0.5
        n_steps = 4
        psi0 = _random_state(n_qubits, seed=11)

        H = sum(c * pauli_hamiltonian_to_matrix([(1.0, p)], n_qubits) for c, p in terms)
        psi_exact = scipy.linalg.expm(-1j * H * t) @ psi0

        psi_order1 = _run_ops_on_state(psi0, n_qubits, trotter_evolve_ops(terms, t, n_steps, order=1))
        psi_order2 = _run_ops_on_state(psi0, n_qubits, trotter_evolve_ops(terms, t, n_steps, order=2))

        infidelity_1 = 1.0 - statevector_fidelity(psi_order1, psi_exact)
        infidelity_2 = 1.0 - statevector_fidelity(psi_order2, psi_exact)
        assert infidelity_2 < infidelity_1

    def test_order_2_single_term_reduces_to_order_1(self):
        """With one term, forward and backward half-passes both act on
        the same single Pauli string, so order=2 should reduce to a
        single order=1-equivalent full-angle rotation per step (the
        two half-angle rotations on the same axis compose additively)."""
        pauli_dict = {0: 'X', 1: 'Z'}
        terms = [(1.0, pauli_dict)]
        t, n_steps = 0.8, 1

        ops_order1 = trotter_evolve_ops(terms, t, n_steps, order=1)
        ops_order2 = trotter_evolve_ops(terms, t, n_steps, order=2)

        n_qubits = 2
        psi0 = _random_state(n_qubits, seed=13)
        psi1 = _run_ops_on_state(psi0, n_qubits, ops_order1)
        psi2 = _run_ops_on_state(psi0, n_qubits, ops_order2)
        assert statevector_fidelity(psi1, psi2) == pytest.approx(1.0, abs=1e-9)

    def test_order_2_gate_count_is_double_order_1(self):
        terms = [(1.0, {0: 'X'}), (1.0, {0: 'Y', 1: 'Z'})]
        ops1 = trotter_evolve_ops(terms, t=1.0, n_steps=5, order=1)
        ops2 = trotter_evolve_ops(terms, t=1.0, n_steps=5, order=2)
        assert len(ops2) == 2 * len(ops1)
