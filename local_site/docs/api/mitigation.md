# Mitigation (Zero-Noise Extrapolation)

Standard error-mitigation entry points, named the way the field already names them
(Richardson extrapolation, noise factors, zero-noise extrapolation), plus a density-matrix
extension (physical-cone projection, Uhlmann fidelity) and a `jax.jit`-compatible variant
of every entry point.

::: dense_evolution.mitigation

---

**See also**: [`dense_evolution.healing`](healing.md) for the predictive-healing primitives
(`calculate_delta_preemp`) the healing-adapted extrapolation branch is built on, and
[`NoiseModel`](registry.md) for the Kraus-channel noise used to build the noisy ensembles
these functions correct. Full worked example: [Density-matrix ZNE healing](../examples.md#density-matrix-zne-healing).
