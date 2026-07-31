"""
Unit tests for dense_evolution/measurement.py -- sample_counts and
statevector_fidelity.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator
from dense_evolution.measurement import sample_counts, statevector_fidelity


class TestSampleCounts:

    def test_counts_sum_to_n_shots(self):
        sim = DenseSVSimulator(2, use_gpu=False, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        counts = sample_counts(sim.get_statevector(), 5000, rng=np.random.default_rng(0))
        assert sum(counts.values()) == 5000

    def test_bell_state_only_populates_00_and_11(self):
        sim = DenseSVSimulator(2, use_gpu=False, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        counts = sample_counts(sim.get_statevector(), 5000, rng=np.random.default_rng(1))
        assert set(counts) <= {'00', '11'}
        # roughly balanced -- generous tolerance, this is a statistical check
        assert 0.4 < counts.get('00', 0) / 5000 < 0.6

    def test_bitstring_matches_deterministic_basis_state(self):
        # X on qubit 1 alone -> deterministic |01>, verifies qubit-0-is-MSB
        # bitstring convention directly (no ambiguity from superposition)
        sim = DenseSVSimulator(2, use_gpu=False, use_float32=False)
        sim.run_circuit([('x', 1)])
        counts = sample_counts(sim.get_statevector(), 10)
        assert counts == {'01': 10}

    def test_rejects_zero_shots(self):
        with pytest.raises(ValueError):
            sample_counts(np.array([1.0, 0, 0, 0]), 0)

    def test_rejects_non_power_of_two_length(self):
        with pytest.raises(ValueError):
            sample_counts(np.array([1.0, 0, 0]), 10)

    def test_reproducible_with_seeded_rng(self):
        sv = np.array([0.6, 0.8], dtype=complex)
        a = sample_counts(sv, 200, rng=np.random.default_rng(42))
        b = sample_counts(sv, 200, rng=np.random.default_rng(42))
        assert a == b


class TestStatevectorFidelity:

    def test_self_fidelity_is_one(self):
        sim = DenseSVSimulator(2, use_gpu=False, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        sv = sim.get_statevector()
        assert statevector_fidelity(sv, sv) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_states_have_zero_fidelity(self):
        a = np.array([1.0, 0, 0, 0], dtype=complex)
        b = np.array([0, 0, 0, 1.0], dtype=complex)
        assert statevector_fidelity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_bell_vs_basis_state_is_half(self):
        sim = DenseSVSimulator(2, use_gpu=False, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        bell = sim.get_statevector()
        basis_11 = np.array([0, 0, 0, 1.0], dtype=complex)
        assert statevector_fidelity(bell, basis_11) == pytest.approx(0.5, abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            statevector_fidelity(np.array([1.0, 0]), np.array([1.0, 0, 0, 0]))
