# Mitigation (Zero-Noise Extrapolation & Density-Matrix Diagnostics)

Standard error-mitigation entry points, named the way the field already names them
(Richardson extrapolation, noise factors, zero-noise extrapolation), plus a density-matrix
extension (physical-cone projection, Uhlmann fidelity) and a `jax.jit`-compatible variant
of every entry point.

::: dense_evolution.mitigation.zne

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
