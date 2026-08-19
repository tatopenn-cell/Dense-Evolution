"""
Unit tests for dense_evolution/mitigation/magic_entropy.py. Values
cross-checked against Dense-Evolution-Discovery Experiment 30
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_ruzsa_magic_entropy/),
where the same numbers were independently validated against the source
paper's own Lemma 9 identity and stabilizer-state claim.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_evolution.mitigation import magic_entropy, magic_entropy_jit
from dense_evolution.mitigation.magic_entropy import _KEY_UNITARY_K3, _self_convolve_3_core


def _sv_to_rho(sv):
    sv = jnp.asarray(sv, dtype=jnp.complex128)
    return jnp.outer(sv, jnp.conj(sv))


def _stabilizer_states():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    plus = (zero + one) / jnp.sqrt(2.0)
    minus = (zero - one) / jnp.sqrt(2.0)
    plus_i = (zero + 1j * one) / jnp.sqrt(2.0)
    minus_i = (zero - 1j * one) / jnp.sqrt(2.0)
    return [zero, one, plus, minus, plus_i, minus_i]


def _t_state():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    return (zero + jnp.exp(1j * jnp.pi / 4.0) * one) / jnp.sqrt(2.0)


def test_key_unitary_matches_paper_lemma_9_identity():
    # V|x1 x2 x3> = |x1+x2+x3> (x) |x2+x1> (x) |x3+x1>  (mod 2)
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                idx_in = (x1 << 2) | (x2 << 1) | x3
                sv_in = jnp.zeros(8, dtype=jnp.complex128).at[idx_in].set(1.0)
                sv_out = _KEY_UNITARY_K3 @ sv_in
                y1, y2, y3 = x1 ^ x2 ^ x3, x2 ^ x1, x3 ^ x1
                idx_expected = (y1 << 2) | (y2 << 1) | y3
                out_idx = int(jnp.argmax(jnp.abs(sv_out)))
                assert out_idx == idx_expected
                assert abs(complex(sv_out[out_idx])) > 0.999


def test_key_unitary_is_unitary():
    v = _KEY_UNITARY_K3
    identity = jnp.eye(8, dtype=jnp.complex128)
    assert np.allclose(np.array(v @ jnp.conj(v).T), np.array(identity), atol=1e-10)


def test_all_single_qubit_stabilizer_states_have_zero_magic_entropy():
    for sv in _stabilizer_states():
        m = magic_entropy(_sv_to_rho(sv))
        assert m < 1e-8


def test_t_and_h_states_match_experiment_30_value():
    t_rho = _sv_to_rho(_t_state())
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    h_sv = jnp.cos(jnp.pi / 8.0) * zero + jnp.sin(jnp.pi / 8.0) * one
    h_rho = _sv_to_rho(h_sv)
    assert magic_entropy(t_rho) == pytest.approx(0.8112781245, abs=1e-6)
    assert magic_entropy(h_rho) == pytest.approx(0.8112781245, abs=1e-6)


def test_fully_mixed_state_has_maximal_magic_entropy():
    mixed = jnp.eye(2, dtype=jnp.complex128) / 2.0
    assert magic_entropy(mixed) == pytest.approx(1.0, abs=1e-6)


def test_jit_matches_eager():
    rho = _sv_to_rho(_t_state())
    eager = magic_entropy(rho)
    jitted = float(magic_entropy_jit(rho))
    assert eager == pytest.approx(jitted, abs=1e-12)


def test_self_convolve_3_output_is_a_valid_density_matrix():
    rho = _sv_to_rho(_t_state())
    reduced = _self_convolve_3_core(rho)
    assert abs(complex(jnp.trace(reduced)) - 1.0) < 1e-9
    ev = np.linalg.eigvalsh(np.array(reduced))
    assert np.all(ev > -1e-9)


def test_differentiable_at_a_degenerate_stabilizer_state():
    # I/2's reduced matrix has exactly degenerate eigenvalues [0.5, 0.5] --
    # the case uhlmann_fidelity needed a custom eigh JVP rule for, since it
    # reconstructs eigenvectors. magic_entropy only needs eigenvalues
    # (jnp.linalg.eigvalsh), whose gradient stays well-defined here.
    def loss(theta):
        # a state that is exactly |0> at theta=0 (a stabilizer state, whose
        # magic-entropy reduced matrix is degenerate) but still lets grad
        # flow through theta.
        sv = jnp.array([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.complex128)
        rho = jnp.outer(sv, jnp.conj(sv))
        from dense_evolution.mitigation.magic_entropy import _magic_entropy_core
        return _magic_entropy_core(rho)

    g = jax.grad(loss)(0.0)
    assert not jnp.isnan(g)


def test_differentiable_at_a_magic_state():
    def loss(theta):
        sv = jnp.array([jnp.cos(theta), jnp.exp(1j * jnp.pi / 4.0) * jnp.sin(theta)], dtype=jnp.complex128)
        rho = jnp.outer(sv, jnp.conj(sv))
        from dense_evolution.mitigation.magic_entropy import _magic_entropy_core
        return _magic_entropy_core(rho)

    # theta=pi/4 is exactly the T-state -- a stationary point of
    # magic_entropy(theta) under this parametrization by symmetry, so its
    # gradient is genuinely 0, not a differentiability failure. Use a
    # generic angle instead to check the gradient is actually informative.
    g = jax.grad(loss)(0.3)
    assert not jnp.isnan(g)
    assert abs(g) > 1e-8
