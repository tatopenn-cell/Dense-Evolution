"""
Unit tests for dense_evolution/states.py -- ghz_state.
"""
import pytest

from dense_evolution import DenseSVSimulator
from dense_evolution.states import ghz_state


class TestGhzState:

    def test_matches_hand_written_circuit(self):
        assert ghz_state(3) == [('h', 0), ('cx', 0, 1), ('cx', 1, 2)]

    def test_rejects_fewer_than_two_qubits(self):
        with pytest.raises(ValueError):
            ghz_state(1)

    @pytest.mark.parametrize('n_qubits', [2, 3, 4, 5])
    def test_produces_equal_superposition_of_all_zero_and_all_one(self, n_qubits):
        sim = DenseSVSimulator(n_qubits, use_gpu=False, use_float32=False)
        sim.run_circuit(ghz_state(n_qubits))
        probs = sim.get_probabilities()
        dim = 2 ** n_qubits
        assert probs[0] == pytest.approx(0.5, abs=1e-9)
        assert probs[dim - 1] == pytest.approx(0.5, abs=1e-9)
        assert probs.sum() - probs[0] - probs[dim - 1] == pytest.approx(0.0, abs=1e-9)
