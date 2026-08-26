"""Depolarizing channel: K0=sqrt(1-p)I, K1=sqrt(p/3)X, K2=sqrt(p/3)Y,
K3=sqrt(p/3)Z. Isotropic Pauli error -- equiprobable X, Y, Z given that
the channel fires at all.

One fire/no-fire decision per qubit per shot (probability p), then --
only if it fired -- one equiprobable Pauli choice among X/Y/Z. The
1-in-3 choice thresholds are fixed at 1/3 and 2/3 regardless of p: p
only gates whether an error happens at all, not which Pauli it is.
"""
import jax
import jax.numpy as jnp

__all__ = ["apply"]

_THIRD = 1.0 / 3.0


def apply(sv_out, idx_0, idx_1, p, rng, key, is_jax):
    if is_jax:
        key, sk1, sk2 = jax.random.split(key, 3)
        r  = jax.random.uniform(sk1, shape=(), minval=0.0, maxval=1.0)
        ch = jax.random.uniform(sk2, shape=(), minval=0.0, maxval=1.0)
        fire   = r < p
        x_gate = fire & (ch < _THIRD)
        y_gate = fire & (ch >= _THIRD) & (ch < 2.0 * _THIRD)
        z_gate = fire & (ch >= 2.0 * _THIRD)
        v0, v1 = sv_out[idx_0], sv_out[idx_1]
        new_v0 = jnp.where(x_gate,  v1,
                 jnp.where(y_gate, -1j * v1, v0))
        new_v1 = jnp.where(x_gate,  v0,
                 jnp.where(y_gate,  1j * v0,
                 jnp.where(z_gate, -v1, v1)))
        sv_out = sv_out.at[idx_0].set(new_v0)
        sv_out = sv_out.at[idx_1].set(new_v1)
    else:
        r  = rng.random()
        ch = rng.random()
        if r < p:
            if ch < _THIRD:
                sv_out[idx_0], sv_out[idx_1] = sv_out[idx_1].copy(), sv_out[idx_0].copy()
            elif ch < 2.0 * _THIRD:
                v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
                sv_out[idx_0] = -1j * v1
                sv_out[idx_1] =  1j * v0
            else:
                sv_out[idx_1] = -sv_out[idx_1]
    return sv_out, key
