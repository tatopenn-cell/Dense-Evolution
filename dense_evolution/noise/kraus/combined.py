"""Combined channel: depolarizing(p/2) then amplitude_damping(p/3),
applied sequentially on the same qubit -- a worst-case NISQ mixture
(dephasing + relaxation). Reuses `depolarizing.apply` and
`amplitude_damping.apply` directly rather than duplicating their logic,
each called with its own scaled probability."""
from . import depolarizing, amplitude_damping

__all__ = ["apply"]


def apply(sv_out, idx_0, idx_1, p, rng, key, is_jax):
    p_dep = p * 0.5
    p_damp = p * 0.333333
    sv_out, key = depolarizing.apply(sv_out, idx_0, idx_1, p_dep, rng, key, is_jax)
    sv_out, key = amplitude_damping.apply(sv_out, idx_0, idx_1, p_damp, rng, key, is_jax)
    return sv_out, key
