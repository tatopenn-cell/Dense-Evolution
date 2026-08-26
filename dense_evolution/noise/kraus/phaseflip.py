"""Phase-flip channel: K0=sqrt(1-p)I, K1=sqrt(p)Z. Z applied with
probability p, once per qubit per shot: Z|0> = |0> (idx_0 amplitudes
untouched), Z|1> = -|1> (idx_1 amplitudes negated when fired)."""
import jax
import jax.numpy as jnp

__all__ = ["apply"]


def apply(sv_out, idx_0, idx_1, p, rng, key, is_jax):
    if is_jax:
        key, subkey = jax.random.split(key)
        r    = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
        fire = r < p
        v1     = sv_out[idx_1]
        sv_out = sv_out.at[idx_1].set(jnp.where(fire, -v1, v1))
    else:
        r = rng.random()
        if r < p:
            sv_out[idx_1] = -sv_out[idx_1]
    return sv_out, key
