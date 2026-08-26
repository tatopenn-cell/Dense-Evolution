"""One file per Kraus channel: `ideal`, `depolarizing`, `bitflip`,
`phaseflip`, `amplitude_damping`, `combined`. Each module exposes a single
`apply(sv_out, idx_0, idx_1, p, rng, key, is_jax) -> (sv_out, key)`
function; `dense_evolution.noise.kraus_channels.NoiseModel` is the shared
dispatcher (RNG/key setup, per-qubit loop, normalisation) that calls them.
"""
