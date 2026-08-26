"""Bit-flip channel: K0=sqrt(1-p)I, K1=sqrt(p)X. X applied with
probability p, once per qubit per shot, uniformly across the whole
statevector."""
import jax
import jax.numpy as jnp

__all__ = ["apply"]


def apply(sv_out, idx_0, idx_1, p, rng, key, is_jax):
    if is_jax:
        key, subkey = jax.random.split(key)
        r    = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
        fire = r < p
        v0, v1 = sv_out[idx_0], sv_out[idx_1]
        sv_out = sv_out.at[idx_0].set(jnp.where(fire, v1, v0))
        sv_out = sv_out.at[idx_1].set(jnp.where(fire, v0, v1))
    else:
        r = rng.random()
        if r < p:
            sv_out[idx_0], sv_out[idx_1] = sv_out[idx_1].copy(), sv_out[idx_0].copy()
    return sv_out, key
