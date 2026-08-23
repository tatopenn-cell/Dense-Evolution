# Mitigation (Zero-Noise Extrapolation & Density-Matrix Diagnostics)

Standard error-mitigation entry points, named the way the field already names them
(Richardson extrapolation, noise factors, zero-noise extrapolation), plus a density-matrix
extension (physical-cone projection, Uhlmann fidelity) and a `jax.jit`-compatible variant
of every entry point.

::: dense_evolution.mitigation.zne

---

## Standalone density-matrix noise channels

Three CPTP channels usable directly on a density matrix, independent of `NoiseModel`'s
per-qubit gate-noise pipeline (`circuits.registry`). `global_depolarizing_channel` is
symmetric -- it mixes the whole register toward the fully-mixed state as one unit -- promoted
from a real reproduction of arXiv:2608.16716's SPAM model
([Experiment 33](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/germanium_iswap_validation/)).
`amplitude_damping_channel` is the opposite kind of asymmetric: population only ever moves
`|1>`&rarr;`|0>`, never the reverse -- the real signature of T1 decay and of quasiparticle
poisoning, promoted from
[Experiment 34](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cosmic_ray_burst_validation/)'s
reproduction of a real cosmic-ray-induced error burst (arXiv:2104.05219). `cosmic_ray_burst_profile`
is not a channel itself but the time-dependent decay-probability GENERATOR that experiment's
real numbers were extracted into a reusable, parametrized form from -- feed its output straight
to `amplitude_damping_channel` via [`continuous_dissipative_evolve`](trotter.md). Both are
already covered by the `dense_evolution.mitigation.zne` API reference above -- this section
is context, not a duplicate listing.

---

## Density-matrix diagnostics

Two further density-matrix diagnostics, both originated as Colab proposals with real bugs,
fixed and validated in [Dense-Evolution-Discovery](https://github.com/tatopenn-cell/Dense-Evolution-Discovery)
before promotion here: a non-commuting-aware divergence (`sandwiched_renyi_divergence`) and a
single-qubit non-stabilizerness measure (`magic_entropy`). Both are validation-only, like
`uhlmann_fidelity` above -- meant to grade a correction against a known reference state, not to
feed into one.

::: dense_evolution.mitigation.renyi

::: dense_evolution.mitigation.magic_entropy

---

## Shadow-based estimation

A classical-shadows-based estimator for `magic_entropy` above, estimating it from randomized
measurement snapshots instead of the exact density matrix. Different API shape from everything
else on this page -- sampling (`sample_classical_shadow`) and estimation
(`magic_entropy_from_shadows`) are separate steps, since shadow data can come from this
simulator's own Born-rule oracle sampling or, in principle, real hardware measurement logs
reconstructed the same way. Not `jax.jit`-compatible (median-of-means uses `numpy.median`).

::: dense_evolution.mitigation.magic_entropy_shadows

---

## Classical distribution divergence

The classical Kullback-Leibler divergence over probability distributions (Kullback & Leibler,
1951) -- distinct from `sandwiched_renyi_divergence` above, which operates on density
*matrices* via matrix logarithms; this operates directly on probability *vectors* (e.g. a
measurement-outcome distribution `jnp.abs(psi) ** 2`), no eigendecomposition needed. Additive
to [`dense_evolution.healing`](healing.md)'s existing scalar log-ratio signal, not a
replacement for it -- validated in
[Dense-Evolution-Discovery, Experiment 32](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kullback_leibler_divergence/)
to be a genuinely different signal on the same states, not a rescaling.

::: dense_evolution.mitigation.kl_divergence

---

**See also**: [`dense_evolution.healing`](healing.md) for the predictive-healing primitives
(`calculate_delta_preemp`) the healing-adapted extrapolation branch is built on, and
[`NoiseModel`](registry.md) for the Kraus-channel noise used to build the noisy ensembles
these functions correct. Full worked example: [Density-matrix ZNE healing](../examples.md#density-matrix-zne-healing).
