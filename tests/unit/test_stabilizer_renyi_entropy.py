"""
Unit tests for dense_evolution/mitigation/stabilizer_renyi_entropy.py.
Known-value checks cross-referenced against Dense-Evolution-Discovery's
wormhole_magic_entropy.py, where the T-state value was independently
derived by hand from the source paper's own Eq. 5 (arXiv:2106.12587), not
copied from anywhere else.
"""
import jax
jax.config.update("jax_enable_x64", True)  # needed for the tight "exactly
# 0 for stabilizer states" tolerances below -- float32 (the default
# without this) was measured to accumulate ~3e-7 error on a 4-qubit GHZ
# state through this module's O(d^3) computation, same pattern already
# used in tests/unit/test_amplitude_damping_channel.py for the same reason.

import numpy as np
import pytest

from dense_evolution.mitigation import stabilizer_renyi_entropy, stabilizer_renyi_entropy_jit
from dense_evolution import DenseSVSimulator


def _original_loop_reference(psi):
    """Independent reference: the original, unvectorized Discovery
    implementation (explicit Python loop over `a`, one (d,d)@(d,) matvec
    per iteration) -- the promoted version replaces this loop with one
    (d,d)@(d,d) matmul; this checks they compute the identical number."""
    psi = np.asarray(psi, dtype=complex)
    d = len(psi)
    idx = np.arange(d)
    popcount = np.array([bin(i).count("1") for i in range(d)])
    signmat = np.array([(-1.0) ** popcount[idx & b] for b in range(d)])
    total = 0.0
    for a in range(d):
        c_a = np.conj(psi) * psi[idx ^ a]
        wht = signmat @ c_a
        total += np.sum(np.abs(wht) ** 4)
    return -np.log2(total / d)


class TestKnownValues:

    def test_computational_basis_state_is_a_stabilizer_state(self):
        for n_qubits in [1, 2, 3, 4]:
            d = 2 ** n_qubits
            for basis_idx in [0, d - 1]:
                psi = np.zeros(d, dtype=complex)
                psi[basis_idx] = 1.0
                assert stabilizer_renyi_entropy(psi) == pytest.approx(0.0, abs=1e-8)

    def test_ghz_state_is_a_stabilizer_state(self):
        for n_qubits in [2, 3, 4]:
            d = 2 ** n_qubits
            psi = np.zeros(d, dtype=complex)
            psi[0] = psi[-1] = 1.0 / np.sqrt(2.0)
            assert stabilizer_renyi_entropy(psi) == pytest.approx(0.0, abs=1e-8)

    def test_bell_state_from_a_real_circuit_is_a_stabilizer_state(self):
        sim = DenseSVSimulator(2, use_float32=False)
        sim.run_circuit([('h', 0), ('cx', 0, 1)])
        psi = sim.get_statevector()
        assert stabilizer_renyi_entropy(psi) == pytest.approx(0.0, abs=1e-8)

    def test_single_t_state_matches_hand_derived_value(self):
        theta = np.pi / 8
        t1 = np.array([np.cos(theta), np.sin(theta)], dtype=complex)
        assert stabilizer_renyi_entropy(t1) == pytest.approx(-np.log2(0.75), abs=1e-6)


class TestVectorizationMatchesOriginalLoop:
    """The promoted implementation replaced an explicit Python loop with
    one batched matmul -- these confirm that was a pure vectorization,
    not a change in the actual computed quantity."""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_matches_original_loop_on_random_states(self, n_qubits):
        rng = np.random.default_rng(42 + n_qubits)
        d = 2 ** n_qubits
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        v /= np.linalg.norm(v)
        assert stabilizer_renyi_entropy(v) == pytest.approx(_original_loop_reference(v), abs=1e-9)


class TestJitAndValidation:

    def test_jit_matches_eager(self):
        theta = np.pi / 8
        t1 = np.array([np.cos(theta), np.sin(theta)], dtype=complex)
        eager = stabilizer_renyi_entropy(t1)
        jitted = float(stabilizer_renyi_entropy_jit(np.asarray(t1, dtype=np.complex128)))
        assert eager == pytest.approx(jitted, abs=1e-10)

    def test_non_power_of_two_length_raises(self):
        with pytest.raises(ValueError):
            stabilizer_renyi_entropy(np.ones(5, dtype=complex) / np.sqrt(5))

    def test_non_negative_for_random_states(self):
        # A Renyi entropy of a valid probability-like distribution can't
        # be negative -- sanity bound, not a known exact value.
        rng = np.random.default_rng(0)
        for n_qubits in [1, 2, 3]:
            d = 2 ** n_qubits
            for _ in range(10):
                v = rng.normal(size=d) + 1j * rng.normal(size=d)
                v /= np.linalg.norm(v)
                assert stabilizer_renyi_entropy(v) >= -1e-9
