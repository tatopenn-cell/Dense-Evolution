"""
Unit tests for dense_evolution/healing.py -- the predictive-healing engine
(calculate_advanced_sigma, calculate_phi_ab, calculate_vettore_dinamico/
statico, calculate_delta_preemp, evaluate_phi_trigger,
calculate_jax_reflection) and MemoryReflectionEngine.

Not to be confused with test_ia_healing.py, which covers a different
module (ia_utils/vector_healing.py).

Split out of the original monolithic test_dense_evolution.py -- see
test_simulator.py's module docstring for why. Audit finding #4: this
module had 0% test coverage. Values below were hand-verified
(independently, before writing assertions) against the actual function
output, matching the audit's methodology of confirming behavior rather
than padding a coverage number.
"""
import numpy as np
import pytest
import jax.numpy as jnp

from dense_evolution import DenseSVSimulator, GATES, healing

# ─────────────────────────────────────────────────────────────
# PREDICTIVE HEALING ENGINE
# ─────────────────────────────────────────────────────────────

class TestPredictiveHealingCore:

    def test_advanced_sigma_is_product_of_inputs(self):
        with pytest.warns(DeprecationWarning, match="calculate_advanced_sigma is deprecated"):
            s = healing.calculate_advanced_sigma(
                jnp.array(2.0), jnp.array(3.0), jnp.array(1.0), jnp.array(1.0), jnp.array(1.0))
        assert float(s) == pytest.approx(6.0)

    def test_phi_ab_identical_states_returns_baseline_0_7(self):
        # state_A == state_B -> norm_change ~ 0 -> alignment defaults to 0.0
        # -> semantic_alignment = 0.5; distance_A_B = 0 -> coherence = 1.0
        # phi_ab = 0.5*0.6 + 1.0*0.4 = 0.7
        a = jnp.array([1.0, 0.0])
        phi = healing.calculate_phi_ab(a, a, jnp.array([1.0, 0.0]))
        assert float(phi) == pytest.approx(0.7, abs=1e-6)

    def test_phi_ab_clipped_to_unit_interval(self):
        a = jnp.array([1.0, 0.0])
        b = jnp.array([-1.0, 0.0])
        phi = healing.calculate_phi_ab(a, b, jnp.array([1.0, 0.0]))
        assert 0.0 <= float(phi) <= 1.0

    def test_phi_ab_handles_complex_statevectors_no_crash(self):
        # Regression test: jnp.dot on complex arrays returns a complex
        # scalar, which used to blow up jnp.clip with "ValueError: Clip
        # received a complex value". Repro from the bug report.
        sim = DenseSVSimulator(n_qubits=3)
        sim.apply_gate_1q(GATES['h'], 0)
        sim.apply_gate_2q(GATES['cx'], 0, 1)
        sv = jnp.array(sim.get_statevector())
        assert jnp.iscomplexobj(sv)
        phi = healing.calculate_phi_ab(sv, sv, sv)
        assert not jnp.iscomplexobj(phi)
        assert 0.0 <= float(phi) <= 1.0

    def test_phi_ab_complex_alignment_matches_manual_hermitian_dot(self):
        # The bug-report repro (H+CX) never produces nonzero imaginary
        # amplitudes, so it only proves "doesn't crash on complex dtype" --
        # this exercises a genuinely complex case (S = phase gate) and
        # cross-checks the result against a plain-NumPy Re(vdot(...)) calc.
        sim_a = DenseSVSimulator(n_qubits=1)
        sim_a.apply_gate_1q(GATES['h'], 0)
        state_A = jnp.array(sim_a.get_statevector())

        sim_b = DenseSVSimulator(n_qubits=1)
        sim_b.apply_gate_1q(GATES['h'], 0)
        sim_b.apply_gate_1q(GATES['s'], 0)
        state_B = jnp.array(sim_b.get_statevector())
        assert np.any(np.abs(np.imag(np.array(state_B))) > 1e-9)  # sanity: genuinely complex

        phi = healing.calculate_phi_ab(state_A, state_B, state_B)
        assert 0.0 <= float(phi) <= 1.0

        sc = np.array(state_B) - np.array(state_A)
        expected_alignment = np.real(np.vdot(sc, np.array(state_B))) / (
            np.linalg.norm(sc) * np.linalg.norm(state_B))
        expected_semantic_alignment = (expected_alignment + 1.0) / 2.0
        dist = np.linalg.norm(np.array(state_A) - np.array(state_B))
        expected_coherence = 1.0 - dist / float(healing.GLOBAL_CONSTANTS['MAX_SEMANTIC_DISTANCE'])
        expected_phi = np.clip(
            expected_semantic_alignment * healing.GLOBAL_CONSTANTS['WEIGHT_SEMANTIC']
            + expected_coherence * healing.GLOBAL_CONSTANTS['WEIGHT_COHERENCE'], 0.0, 1.0)
        assert float(phi) == pytest.approx(float(expected_phi), abs=1e-9)

    def test_vettore_dinamico_zero_energy_is_guarded(self):
        # E_A=0 must hit the invalid_inputs branch and return 0.0, not
        # propagate a log(inf)/NaN from the ratio computation.
        vd = healing.calculate_vettore_dinamico(jnp.array(0.0), jnp.array(5.0), jnp.array(0.7))
        assert float(vd) == pytest.approx(0.0)
        assert not np.isnan(float(vd))

    def test_vettore_dinamico_equal_energies_is_zero(self):
        vd = healing.calculate_vettore_dinamico(jnp.array(5.0), jnp.array(5.0), jnp.array(0.7))
        assert float(vd) == pytest.approx(0.0, abs=1e-6)

    def test_vettore_statico_growing_vs_static_branches(self):
        growing = healing.calculate_vettore_statico(jnp.array(0.5))   # > MIN_EFFECTIVE_VALUE (0.01)
        static = healing.calculate_vettore_statico(jnp.array(0.001))  # < MIN_EFFECTIVE_VALUE
        assert float(growing) == pytest.approx(0.0)
        assert float(static) == pytest.approx(1.0)

    def test_delta_preemp_zero_deviation(self):
        d = healing.calculate_delta_preemp(jnp.array(10.0), 10.0)
        assert float(d) == pytest.approx(0.0)

    def test_delta_preemp_half_deviation(self):
        d = healing.calculate_delta_preemp(jnp.array(5.0), 10.0)
        assert float(d) == pytest.approx(0.5)

    def test_delta_preemp_nonpositive_target_falls_back_to_1(self):
        # target_sigma_ideal <= 0 -> safe_target = 1.0, avoids division by <=0
        d = healing.calculate_delta_preemp(jnp.array(5.0), -1.0)
        assert float(d) == pytest.approx(6.0)  # |5 - (-1)| / 1.0

    def test_phi_trigger_active_branch(self):
        trigger, lam, eps = healing.evaluate_phi_trigger(jnp.array(0.5))  # > NON_STATIC_THRESHOLD_A (1e-2)
        assert float(trigger) == pytest.approx(1.0)
        assert float(lam) == pytest.approx(0.05)
        assert float(eps) == pytest.approx(0.01)

    def test_phi_trigger_stasis_branch(self):
        trigger, lam, eps = healing.evaluate_phi_trigger(jnp.array(0.001))  # < threshold
        assert float(trigger) == pytest.approx(0.0)
        assert float(lam) == pytest.approx(0.15)
        assert float(eps) == pytest.approx(0.11)

    def test_jax_reflection_empty_arrays_returns_zero_not_nan(self):
        # calculate_jax_reflection guards n==0 via jnp.where; confirms the
        # guard actually suppresses NaN from the discarded jnp.mean/var
        # branch rather than letting it leak through the select.
        avg_c, var_c, avg_n = healing.calculate_jax_reflection(
            jnp.array([], dtype=jnp.float64), jnp.array([], dtype=jnp.float64))
        assert float(avg_c) == 0.0 and not np.isnan(float(avg_c))
        assert float(var_c) == 0.0 and not np.isnan(float(var_c))
        assert float(avg_n) == 0.0 and not np.isnan(float(avg_n))

    def test_jax_reflection_nonempty_matches_plain_numpy(self):
        coh = jnp.array([0.8, 0.6, 0.9], dtype=jnp.float64)
        noise = jnp.array([0.1, 0.2], dtype=jnp.float64)
        avg_c, var_c, avg_n = healing.calculate_jax_reflection(coh, noise)
        assert float(avg_c) == pytest.approx(np.mean([0.8, 0.6, 0.9]))
        assert float(var_c) == pytest.approx(np.var([0.8, 0.6, 0.9]))
        assert float(avg_n) == pytest.approx(np.mean([0.1, 0.2]))


class TestMemoryReflectionEngine:

    def test_reflect_aggregates_events_correctly(self):
        eng = healing.MemoryReflectionEngine()
        eng.record_event('coherence', 0.8, 'run1')
        eng.record_event('coherence', 0.6, 'run2')
        eng.record_event('noise', 0.1, 'run1')
        eng.record_event('intervention', 1.0, 'fallback triggered')

        report = eng.reflect()
        assert report['average_coherence'] == pytest.approx(0.7)
        assert report['variance_coherence'] == pytest.approx(np.var([0.8, 0.6]))
        assert report['interventions_count'] == 1
        assert report['average_noise'] == pytest.approx(0.1)
        assert report['total_events'] == 4

    def test_reflect_on_empty_engine_returns_none_not_nan(self):
        eng = healing.MemoryReflectionEngine()
        report = eng.reflect()
        assert report == {
            'average_coherence': None,
            'variance_coherence': None,
            'interventions_count': 0,
            'average_noise': None,
            'total_events': 0,
        }

    def test_export_and_load_memory_round_trip(self, tmp_path):
        eng = healing.MemoryReflectionEngine()
        eng.record_event('coherence', 0.8, 'run1')
        eng.record_event('noise', 0.1, 'run1')

        path = tmp_path / "memory.json"
        eng.export_memory(str(path))

        eng2 = healing.MemoryReflectionEngine()
        eng2.load_memory(str(path))

        assert eng2.memory == eng.memory
        assert eng2.reflect() == eng.reflect()
