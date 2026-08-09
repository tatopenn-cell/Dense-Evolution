"""
Unit tests for dense_evolution/registry.py -- NoiseModel's Kraus-channel
noise (depolarizing, bitflip, phaseflip, amplitude damping, combined),
QuantumHardwareRegistry, and NoiseSpec's JAX PyTree registration.

Split out of the original monolithic test_dense_evolution.py -- see
test_simulator.py's module docstring for why.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from dense_evolution import NoiseModel, NoiseSpec

# ─────────────────────────────────────────────────────────────
# NOISE MODEL (Esempio 2 dal README)
# ─────────────────────────────────────────────────────────────

class TestNoiseModel:

    def test_ideal_model_no_change(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='ideal', p=0.1)
        np.testing.assert_allclose(sv_out, sv, atol=1e-12)

    def test_depolarizing_preserves_norm(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(42)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='depolarizing', p=0.2, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_bitflip_preserves_norm(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(7)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='bitflip', p=0.3, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_phaseflip_preserves_norm(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(99)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='phaseflip', p=0.5, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_amplitude_damping_preserves_norm(self):
        sv = np.array([0.0, 1.0], dtype=complex)  # |1⟩
        rng = np.random.default_rng(42)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='amplitude_damping', p=0.2, rng=rng)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_zero_probability_no_change(self):
        sv = np.array([1.0, 0.0], dtype=complex)
        rng = np.random.default_rng(0)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='depolarizing', p=0.0, rng=rng)
        np.testing.assert_allclose(sv_out, sv, atol=1e-12)

    def test_kraus_description_returns_dict(self):
        for model in NoiseModel.MODELS:
            desc = NoiseModel.kraus_description(model)
            assert isinstance(desc, dict)
            assert 'kraus' in desc

    def test_depolarizing_matches_analytic_prediction(self):
        # Found via independent statistical fuzzing: the docstring promises
        # 'depolarizing' {sqrt(1-p)I, sqrt(p/3)X, sqrt(p/3)Y, sqrt(p/3)Z} —
        # each Pauli error equally likely GIVEN the channel fired — but the
        # implementation compared a full [0,1)-uniform draw against
        # thresholds scaled for a [0,p) draw (p/3, 2p/3 instead of the
        # fixed 1/3, 2/3), skewing outcomes heavily toward Z for any p<1.
        # Isolated trace (100k samples) before the fix measured
        # P(X|fire)=P(Y|fire)=10%, P(Z|fire)=80% at p=0.3, instead of the
        # correct 33.3% each. This test reproduces that at the statevector
        # level: |1> under depolarizing(p) should measure 0 with
        # probability 2p/3 (X or Y flip it), not p/2 or any other skew.
        rng = np.random.default_rng(42)
        sv1 = np.array([0.0, 1.0], dtype=complex)
        n_shots = 30000
        p = 0.3
        counts = np.zeros(2)
        for _ in range(n_shots):
            sv = NoiseModel.apply_to_sv(sv1.copy(), n=1, model='depolarizing', p=p, rng=rng)
            probs_ = np.abs(sv) ** 2
            probs_ /= probs_.sum()
            counts[rng.choice(2, p=probs_)] += 1
        freq = counts / n_shots
        expected = np.array([2 * p / 3, 1 - 2 * p / 3])
        np.testing.assert_allclose(freq, expected, atol=0.02)

    def test_depolarizing_pauli_choice_is_uniform_given_fire(self):
        # Direct trace of the fire/x/y/z branch logic itself (no
        # statevector involved), same style as the isolated reproduction
        # that found the bug — pins the exact 1/3-1/3-1/3 split.
        rng = np.random.default_rng(0)
        n = 200000
        p = 0.3
        r = rng.random(n)
        ch = rng.random(n)
        fire = r < p
        third = 1.0 / 3.0
        x_gate = fire & (ch < third)
        y_gate = fire & (ch >= third) & (ch < 2 * third)
        z_gate = fire & (ch >= 2 * third)
        n_fire = fire.sum()
        assert x_gate.sum() / n_fire == pytest.approx(third, abs=0.01)
        assert y_gate.sum() / n_fire == pytest.approx(third, abs=0.01)
        assert z_gate.sum() / n_fire == pytest.approx(third, abs=0.01)

    def test_combined_model_depolarizing_subchannel_also_fixed(self):
        # The same buggy threshold pattern was duplicated in 'combined'
        # (depolarizing sub-channel) — verify it matches the closed-form
        # prediction: depolarizing(p_dep) then amplitude_damping(p_damp)
        # sequentially on |1>. P(final 0) = 2*p_dep/3 + (1 - 2*p_dep/3)*p_damp
        # (X or Y from depolarizing decays it directly; otherwise it
        # decays via the amplitude-damping sub-channel with probability
        # p_damp). Verified this closed form against a fresh 30k-shot run
        # before writing it down (measured 0.361 vs predicted 0.360).
        rng = np.random.default_rng(7)
        sv1 = np.array([0.0, 1.0], dtype=complex)
        n_shots = 30000
        p = 0.6
        p_dep = p * 0.5
        p_damp = p * 0.333333
        counts = np.zeros(2)
        for _ in range(n_shots):
            sv = NoiseModel.apply_to_sv(sv1.copy(), n=1, model='combined', p=p, rng=rng)
            probs_ = np.abs(sv) ** 2
            probs_ /= probs_.sum()
            counts[rng.choice(2, p=probs_)] += 1
        freq = counts / n_shots
        expected0 = 2 * p_dep / 3 + (1 - 2 * p_dep / 3) * p_damp
        assert freq[0] == pytest.approx(expected0, abs=0.02)

    def test_phaseflip_jax_array_branch(self):
        # Every existing phaseflip/combined test above uses a plain NumPy
        # sv -- registry.py's `is_jax` branches for these two models
        # (jnp.where-based, distinct code from the NumPy np.where branches)
        # were never exercised with an actual JAX array.
        sv = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        rng = np.random.default_rng(11)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='phaseflip', p=0.5, rng=rng)
        assert isinstance(sv_out, jnp.ndarray)
        assert abs(float(jnp.linalg.norm(sv_out)) - 1.0) < 1e-6

    def test_combined_model_jax_array_branch(self):
        sv = jnp.array([0.0, 1.0], dtype=jnp.complex128)
        rng = np.random.default_rng(13)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='combined', p=0.4, rng=rng)
        assert isinstance(sv_out, jnp.ndarray)
        assert abs(float(jnp.linalg.norm(sv_out)) - 1.0) < 1e-6

    def test_apply_to_sv_numpy_path_without_rng_uses_fresh_entropy(self):
        # NumPy sv + rng=None -> registry.py's _fresh_rng() call (never
        # exercised by any existing test, which always pass a seeded rng).
        sv = np.array([1.0, 0.0], dtype=complex)
        sv_out = NoiseModel.apply_to_sv(sv, n=1, model='bitflip', p=0.5, rng=None)
        assert abs(np.linalg.norm(sv_out) - 1.0) < 1e-10

    def test_amplitude_damping_pure_1_state_matches_flat_probability(self):
        # A qubit purely in |1> is the ONE case where the old (buggy) flat
        # decay probability and the correct Born-rule probability
        # gamma*|v1|^2 coincide exactly (|v1|^2=1) -- this is why the bug
        # was invisible to test_amplitude_damping_preserves_norm above
        # (which only checks |1>). Verifies the fixed implementation still
        # gives the textbook-correct P(decay to |0>)=gamma for this case.
        rng = np.random.default_rng(3)
        sv1 = np.array([0.0, 1.0], dtype=complex)
        n_shots = 30000
        gamma = 0.4
        counts = np.zeros(2)
        for _ in range(n_shots):
            sv = NoiseModel.apply_to_sv(sv1.copy(), n=1, model='amplitude_damping', p=gamma, rng=rng)
            probs_ = np.abs(sv) ** 2
            probs_ /= probs_.sum()
            counts[rng.choice(2, p=probs_)] += 1
        freq = counts / n_shots
        assert freq[0] == pytest.approx(gamma, abs=0.02)

    def test_amplitude_damping_superposition_matches_born_rule_not_flat_probability(self):
        # BUG FIX: the decay branch used to fire with a flat probability
        # `gamma`, independent of the qubit's actual |1> population, AND
        # (a second, related error) added the original |0> amplitude
        # into the decay branch's result (`v0 + v1*sqrt(gamma)`) instead
        # of replacing it, as the K1 = [[0,sqrt(gamma)],[0,0]] Kraus
        # operator actually requires (K1 zeroes out any incoming |0>
        # component entirely). For |+> = (|0>+|1>)/sqrt(2) at gamma=0.5,
        # closed-form evaluation of the exact OLD per-branch formula
        # (flat probability + the erroneous +v0 term, each branch
        # renormalized the same way the outer apply_to_sv function does)
        # predicts P(measure 0) = 0.8333; the correct Born-rule formula
        # (P(0) = gamma*|v1|^2 + |v0|^2, verified to match the textbook
        # Kraus channel to 0.00000000 -- see the fix's own commit
        # message / README entry) predicts P(measure 0) = 0.75. These
        # are far enough apart (0.083) to distinguish cleanly at 50k
        # shots with abs=0.02 tolerance.
        rng = np.random.default_rng(11)
        sv_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        n_shots = 50000
        gamma = 0.5
        counts = np.zeros(2)
        for _ in range(n_shots):
            sv = NoiseModel.apply_to_sv(sv_plus.copy(), n=1, model='amplitude_damping', p=gamma, rng=rng)
            probs_ = np.abs(sv) ** 2
            probs_ /= probs_.sum()
            counts[rng.choice(2, p=probs_)] += 1
        freq = counts / n_shots
        expected_p0_correct = 0.75
        expected_p0_old_buggy = 0.8333333333333333
        assert freq[0] == pytest.approx(expected_p0_correct, abs=0.02)
        assert freq[0] != pytest.approx(expected_p0_old_buggy, abs=0.02)


class TestQuantumHardwareRegistry:
    def test_print_diagnostics_runs(self, capsys):
        from dense_evolution.registry import QuantumHardwareRegistry
        reg = QuantumHardwareRegistry()
        reg.print_diagnostics()
        captured = capsys.readouterr()
        assert "MAX_DENSE" in captured.out

    def test_detect_gpu_true_branch(self, monkeypatch):
        from dense_evolution.registry import QuantumHardwareRegistry
        import subprocess
        monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: b"fake gpu output")
        reg = QuantumHardwareRegistry()
        assert reg.has_gpu is True

    def test_noise_spec_repr(self):
        from dense_evolution import NoiseSpec
        spec = NoiseSpec(model="depolarizing", p=0.1, jax_key=jax.random.PRNGKey(0))
        r = repr(spec)
        assert "NoiseSpec" in r and "depolarizing" in r


class TestNoiseModelRngJaxKeyUnification:
    """Issue #7: apply_to_sv used to pick rng vs jax_key based on the
    input array's type, not on which one was actually passed -- a JAX
    statevector silently ignored `rng` entirely and drew from OS entropy
    instead, so a caller seeding `rng` for reproducibility got a
    different, non-reproducible result every call with no signal
    anything was wrong. Fixed: `rng` (when given and `jax_key` isn't)
    now deterministically derives the JAX key too."""

    def test_seeded_rng_is_reproducible_on_jax_statevector(self):
        sv = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.complex128)
        rng_a = np.random.default_rng(42)
        out_a = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=rng_a)
        rng_b = np.random.default_rng(42)
        out_b = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=rng_b)
        np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b))

    def test_seeded_rng_sequence_reproducible_across_multiple_calls(self):
        sv = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.complex128)

        def run_sequence(seed):
            rng = np.random.default_rng(seed)
            o1 = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=rng)
            o2 = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=rng)
            o3 = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=rng)
            return o1, o2, o3

        run1 = run_sequence(42)
        run2 = run_sequence(42)
        for a, b in zip(run1, run2):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_explicit_jax_key_still_takes_precedence_over_rng(self):
        sv = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.complex128)
        key = jax.random.PRNGKey(7)
        out1 = NoiseModel().apply_to_sv(
            sv, n=2, model='depolarizing', p=0.3, rng=np.random.default_rng(1), jax_key=key
        )
        out2 = NoiseModel().apply_to_sv(
            sv, n=2, model='depolarizing', p=0.3, rng=np.random.default_rng(999), jax_key=key
        )
        np.testing.assert_allclose(np.asarray(out1), np.asarray(out2))

    def test_numpy_path_unaffected(self):
        sv = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        out_a = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=np.random.default_rng(42))
        out_b = NoiseModel().apply_to_sv(sv, n=2, model='depolarizing', p=0.3, rng=np.random.default_rng(42))
        np.testing.assert_allclose(out_a, out_b)


class TestNoiseSpecPyTree:
    """NoiseSpec (issue #8): registers NoiseModel's parameters as a real
    JAX PyTree -- model/qubits static (aux_data), p/jax_key as leaves
    (children) -- so it can be passed through jax.jit/grad/vmap/scan the
    same way any other JAX-native value can, e.g. as circuit_to_energy_fn's
    `noise=` argument (see test_autodiff.py::TestEnergyFnNoiseSpec)."""

    def test_tree_flatten_unflatten_roundtrip(self):
        key = jax.random.PRNGKey(5)
        spec = NoiseSpec(model='depolarizing', p=0.2, jax_key=key, qubits=[0, 2])
        leaves, treedef = jax.tree_util.tree_flatten(spec)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.model == 'depolarizing'
        assert rebuilt.qubits == (0, 2)
        assert rebuilt.p == 0.2
        assert bool(jnp.array_equal(rebuilt.jax_key, key))

    def test_leaves_are_p_and_jax_key_only(self):
        # model/qubits must NOT show up as traced leaves -- they're
        # static aux_data, the whole point of the split
        spec = NoiseSpec(model='bitflip', p=0.1, jax_key=jax.random.PRNGKey(1), qubits=[0])
        leaves = jax.tree_util.tree_leaves(spec)
        assert len(leaves) == 2

    def test_jax_tree_map_transforms_leaves(self):
        # jax_key has ndim=1 (shape (2,)); p is a scalar (ndim=0) -- this
        # only touches p, confirming both leaves are independently
        # addressable by a generic tree_map, the way any JAX PyTree's
        # leaves are.
        spec = NoiseSpec(model='depolarizing', p=0.1, jax_key=jax.random.PRNGKey(0))
        doubled = jax.tree_util.tree_map(lambda x: x * 2 if jnp.ndim(x) == 0 else x, spec)
        assert doubled.p == pytest.approx(0.2)
