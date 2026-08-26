"""Amplitude-damping channel: K0=diag(1, sqrt(1-gamma)), K1=[[0,
sqrt(gamma)], [0, 0]]. T1 energy relaxation |1> -> |0> with rate gamma.

Single-trajectory ("quantum jump") unraveling: ONE decay/no-decay
decision per qubit per shot, using the Born-rule probability aggregated
over the WHOLE statevector, P(K1) = <psi|K1^dagger K1|psi> =
gamma * sum_i |v1[i]|^2 (summed over every branch of the other n-1
qubits, not per branch). If it fires, K1 collapses the entire qubit-q=1
branch onto q=0, preserving the other qubits' relative amplitudes
(v1[i]/norm, not flattened to unit phase per branch). Renormalizing by
the GLOBAL P(K1) (or 1-P(K1) for the no-decay branch), not a per-branch
norm, is what a correct sequential multi-qubit trajectory needs.
"""
import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["apply"]


def apply(sv_out, idx_0, idx_1, p, rng, key, is_jax):
    gamma = float(np.clip(p, 0.0, 1.0))
    if is_jax:
        key, subkey = jax.random.split(key)
        r = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
        v0, v1 = sv_out[idx_0], sv_out[idx_1]
        p1 = jnp.clip(gamma * jnp.sum(jnp.abs(v1) ** 2), 0.0, 1.0)
        decay = r < p1
        norm_decay    = jnp.sqrt(jnp.maximum(p1, 1e-15))
        norm_no_decay = jnp.sqrt(jnp.maximum(1.0 - p1, 1e-15))
        new_v0 = jnp.where(decay, v1 * jnp.sqrt(gamma) / norm_decay, v0 / norm_no_decay)
        new_v1 = jnp.where(decay, 0.0 + 0j, v1 * jnp.sqrt(1.0 - gamma) / norm_no_decay)
        sv_out = sv_out.at[idx_0].set(new_v0)
        sv_out = sv_out.at[idx_1].set(new_v1)
    else:
        r = rng.random()
        v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
        p1 = float(np.clip(gamma * np.sum(np.abs(v1) ** 2), 0.0, 1.0))
        if r < p1:
            norm_decay = np.sqrt(max(p1, 1e-15))
            sv_out[idx_0] = v1 * np.sqrt(gamma) / norm_decay
            sv_out[idx_1] = 0.0
        else:
            norm_no_decay = np.sqrt(max(1.0 - p1, 1e-15))
            sv_out[idx_0] = v0 / norm_no_decay
            sv_out[idx_1] = v1 * np.sqrt(1.0 - gamma) / norm_no_decay
    return sv_out, key
