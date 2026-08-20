"""
Unit tests for dense_evolution/random_circuit.py.
"""
import pytest

from dense_evolution import DenseSVSimulator
from dense_evolution.circuits.random_circuit import random_circuit


class TestRandomCircuit:

    def test_generates_requested_number_of_gates(self):
        assert len(random_circuit(4, 25, seed=1)) == 25

    def test_zero_gates_returns_empty_list(self):
        assert random_circuit(3, 0, seed=1) == []

    def test_reproducible_with_same_seed(self):
        a = random_circuit(4, 30, seed=42)
        b = random_circuit(4, 30, seed=42)
        assert a == b

    def test_different_seeds_usually_differ(self):
        a = random_circuit(4, 30, seed=1)
        b = random_circuit(4, 30, seed=2)
        assert a != b

    def test_runs_cleanly_on_the_real_simulator(self):
        circuit = random_circuit(5, 50, seed=7)
        sim = DenseSVSimulator(5, use_float32=False)
        sim.run_circuit(circuit)
        probs = sim.get_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_single_qubit_never_emits_two_qubit_gates(self):
        circuit = random_circuit(1, 30, seed=3)
        assert all(len(op) == 2 or op[0] in ('rx', 'ry', 'rz') for op in circuit)
        assert all(op[1] == 0 for op in circuit)

    def test_gate_set_restricts_output(self):
        circuit = random_circuit(4, 40, seed=9, gate_set=['h', 'cx'])
        assert all(op[0] in ('h', 'cx') for op in circuit)

    def test_unknown_gate_in_gate_set_raises(self):
        with pytest.raises(ValueError):
            random_circuit(3, 5, gate_set=['not_a_gate'])

    def test_rejects_fewer_than_one_qubit(self):
        with pytest.raises(ValueError):
            random_circuit(0, 5)

    def test_rejects_negative_n_gates(self):
        with pytest.raises(ValueError):
            random_circuit(3, -1)

    def test_rejects_out_of_range_two_qubit_prob(self):
        with pytest.raises(ValueError):
            random_circuit(3, 5, two_qubit_prob=1.5)
