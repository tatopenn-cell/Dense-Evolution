"""
Unit tests for dense_evolution/topology.py -- entangling_layer's named
connectivity patterns.
"""
import pytest

from dense_evolution.topology import entangling_layer, VALID_PATTERNS


class TestEntanglingLayer:

    def test_linear_chain(self):
        assert entangling_layer(4, pattern='linear') == [
            ('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 3)]

    def test_circular_adds_wraparound_edge(self):
        layer = entangling_layer(4, pattern='circular')
        assert layer == [('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 3), ('cx', 3, 0)]

    def test_circular_with_two_qubits_equals_linear(self):
        # only one possible edge between two qubits -- no separate
        # wraparound edge makes sense
        assert entangling_layer(2, pattern='circular') == entangling_layer(2, pattern='linear')

    def test_full_is_all_pairs(self):
        layer = entangling_layer(4, pattern='full')
        pairs = {(a, b) for _, a, b in layer}
        expected = {(i, j) for i in range(4) for j in range(i + 1, 4)}
        assert pairs == expected
        assert len(layer) == 6  # C(4,2)

    def test_star_connects_hub_to_every_other_qubit(self):
        layer = entangling_layer(5, pattern='star', hub=2)
        pairs = {(a, b) for _, a, b in layer}
        assert pairs == {(2, 0), (2, 1), (2, 3), (2, 4)}

    def test_star_invalid_hub_raises(self):
        with pytest.raises(ValueError):
            entangling_layer(4, pattern='star', hub=9)

    def test_brick_alternates_even_odd_layers(self):
        layer = entangling_layer(5, pattern='brick')
        pairs = [(a, b) for _, a, b in layer]
        assert pairs == [(0, 1), (2, 3), (1, 2), (3, 4)]

    def test_custom_gate_name(self):
        layer = entangling_layer(3, pattern='linear', gate='cz')
        assert all(op[0] == 'cz' for op in layer)

    def test_reverse_swaps_control_and_target(self):
        forward = entangling_layer(3, pattern='linear')
        backward = entangling_layer(3, pattern='linear', reverse=True)
        assert backward == [(g, b, a) for g, a, b in forward]

    def test_rejects_fewer_than_two_qubits(self):
        with pytest.raises(ValueError):
            entangling_layer(1)

    def test_rejects_unknown_pattern(self):
        with pytest.raises(ValueError):
            entangling_layer(3, pattern='hexagonal')

    @pytest.mark.parametrize('pattern', VALID_PATTERNS)
    def test_every_pattern_runs_on_a_real_circuit(self, pattern):
        # entangling_layer's output must be directly usable by run_circuit
        # -- exercise every named pattern end to end, not just as data.
        from dense_evolution import DenseSVSimulator
        sim = DenseSVSimulator(5, use_gpu=False, use_float32=False)
        sim.run_circuit([('h', 0)] + entangling_layer(5, pattern=pattern))
        probs = sim.get_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-9
