# Mitigation (Zero-Noise Extrapolation & Density-Matrix Diagnostics)

> Correcting a *quantum measurement result*, not a numeric log/trajectory — see
> [Concepts](../concepts.md) if you're looking for [Vector Healing](ia_utils_vector_healing.md) instead.

A real circuit run on noisy hardware gives the wrong answer. Zero-Noise Extrapolation
(ZNE) gets closer to the right one without needing a better device: run the *same*
circuit at several deliberately-worsened noise strengths, then extrapolate the trend
back to what zero noise would have given.

## Step 1. Run the circuit you want to correct

```python
import numpy as np
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv0 = np.asarray(sim.get_statevector())
round(float(abs(sv0[0]) ** 2), 4)
```

```
0.5
```

`sv0` is the exact, noiseless Bell state this whole page corrects a noisy version of.
`abs(sv0[0]) ** 2` is the population of `|00>` — exactly 0.5, since a Bell state is an
equal mix of `|00>` and `|11>`. Every step below tries to recover this same number from
noisy data.

## Step 2. The same circuit, run through a noisy channel at increasing strength

```python
from dense_evolution.noise import NoiseModel

scales = (1.0, 2.0, 3.0)
rng = np.random.default_rng(0)
pop00 = [
    np.mean([abs(NoiseModel.apply_to_sv(sv0.copy(), 2, "depolarizing", 0.05 * s, rng=rng)[0]) ** 2
             for _ in range(20000)])
    for s in scales
]
[round(float(x), 4) for x in pop00]
```

```
[0.4671, 0.4379, 0.4127]
```

Each entry is `|00>`'s population averaged over 20000 noisy runs at one noise scale —
1x, 2x, and 3x a base error rate ([`NoiseModel.apply_to_sv`](noise.md), the same
function [Noise](noise.md) introduces). As the scale grows, the noisy population drifts
further from Step 1's ideal 0.5. These three numbers, paired with `scales`, are exactly
what every extrapolation function below expects as input.

## Step 3. Extrapolate back to zero noise

```python
from dense_evolution.mitigation import richardson_extrapolate

round(float(richardson_extrapolate(pop00, scales)), 4)
```

```
0.5003
```

`richardson_extrapolate` fits an exact curve through the 3 noisy points from Step 2 and
reads off its value at noise scale 0 — recovering Step 1's ideal 0.5 to within rounding
error, from data that never included it. This is the core of every other function on
this page; `zero_noise_extrapolation` (below) and `zne_density_matrix` (Step 5) both
call it, or its density-matrix generalization, internally.

## Step 4. More noise scales than 3 — `polynomial_extrapolate`

```python
from dense_evolution.mitigation import polynomial_extrapolate

more_scales = (1.0, 2.0, 3.0, 4.0, 5.0)
more_pop00 = [
    np.mean([abs(NoiseModel.apply_to_sv(sv0.copy(), 2, "depolarizing", 0.05 * s, rng=rng)[0]) ** 2
             for _ in range(20000)])
    for s in more_scales
]
round(float(polynomial_extrapolate(more_pop00, more_scales, degree=2)), 4)
```

```
0.4979
```

`richardson_extrapolate` needs exactly as many points as it has degrees of freedom, so
its fit becomes numerically unstable with many closely-spaced scales. `polynomial_extrapolate`
fits a lower-degree polynomial (`degree=2` here) by least squares instead — at exactly 3
points it's mathematically identical to Step 3, but with extra points it *averages down*
noise instead of forcing an increasingly ill-conditioned exact fit through every one of
them.

## Step 5. A real experiment gives you a density matrix, not one number

```python
import jax.numpy as jnp
from dense_evolution.mitigation import zne_density_matrix

def noisy_rho(p, k, rng):
    rho = np.zeros((4, 4), dtype=np.complex128)
    for _ in range(k):
        s = NoiseModel.apply_to_sv(sv0.copy(), 2, "depolarizing", p, rng=rng)
        rho += np.outer(s, s.conj())
    return rho / k

rho_at_scales = jnp.stack([noisy_rho(0.05 * s, 200, rng) for s in scales])
corrected = zne_density_matrix(rho_at_scales, scales)
round(float(jnp.trace(corrected).real), 6)
```

```
1.0
```

`rho_at_scales[i]` is the full noisy density matrix at scale `scales[i]` — the
density-matrix counterpart of Step 2's `pop00`. Extrapolating a whole matrix the way
Step 4 extrapolates one number does not generally give back a valid density matrix
(negative eigenvalues can appear even though every input matrix was physical);
`zne_density_matrix` runs `polynomial_extrapolate` on the whole matrix and then projects
the result onto the nearest true density matrix (`project_to_physical`, Details below)
so `corrected` is always physical — trace 1, as shown above — even when the raw
extrapolation wasn't.

## Step 6. Grade the correction

```python
from dense_evolution.mitigation import uhlmann_fidelity

rho_ideal = jnp.asarray(np.outer(sv0, sv0.conj()), dtype=jnp.complex128)
raw_fidelity = uhlmann_fidelity(rho_at_scales[0], rho_ideal)
corrected_fidelity = uhlmann_fidelity(corrected, rho_ideal)
round(float(raw_fidelity), 4), round(float(corrected_fidelity), 4)
```

```
(0.915, 0.975)
```

`uhlmann_fidelity` compares a density matrix against a known ideal state — here, the raw
base-scale noisy result from Step 5 against `rho_ideal` (built from Step 1's `sv0`,
never fed into Steps 2-5), versus the same comparison after correction. `rho_ideal` is
only ever used for this final grading step — feeding a known-ideal state into the
extrapolation or projection steps themselves would be oracle access, not error
mitigation.

## Step 7. When the decay is exponential, not polynomial — `bounded_exponential_extrapolate`

```python
from dense_evolution.observables import pauli_expectation
from dense_evolution.mitigation import bounded_exponential_extrapolate

rng = np.random.default_rng(1)
zz_at_scales = [
    np.mean([pauli_expectation(np.asarray(
        NoiseModel.apply_to_sv(sv0.copy(), 2, "depolarizing", 0.25 * s, rng=rng)), "ZZ")
             for _ in range(200)])
    for s in scales
]
[round(float(x), 4) for x in zz_at_scales]
round(float(bounded_exponential_extrapolate(zz_at_scales, list(scales))), 4)
```

```
[0.49, 0.1, 0.1]
1.0
```

At this much higher noise (`0.25 * s` instead of Step 2's `0.05 * s`), `sv0`'s ideal
`ZZ` expectation of `1.0` decays fast enough that an ordinary unconstrained
`a + b*exp(-c*lambda)` fit on these same 3 points doesn't just misfire — it fails to
converge at all (`scipy.optimize.curve_fit` raises `RuntimeError: Optimal parameters
not found`). `bounded_exponential_extrapolate` reparametrizes the same exponential
model so the zero-noise value is an explicit, constrained parameter
(Miranskyy, Sorrenti, Thind & Gravel, [arXiv:2604.24475](https://arxiv.org/abs/2604.24475)) and
recovers the ideal `1.0` exactly. Reach for this instead of Step 3/4's polynomial
fits when the underlying decay is closer to exponential than polynomial — the usual
shape for a single depolarizing-type channel.

---

## Details

### Healing-adapted zero-noise extrapolation

`zero_noise_extrapolation(expectation_values, noise_factors)` is `richardson_extrapolate`
by default (Step 3 above), but accepts an optional `sigma_at_base_noise`: when given
(alongside exactly 3 noise factors — it raises `NotImplementedError` for any other
count), the 3 Richardson coefficients are perturbed by
[`dense_evolution.healing`](healing.md)'s `calculate_delta_preemp` before renormalizing,
nudging the extrapolation when the measured coherence signal is off an ideal target
instead of trusting the 3 raw points equally.

### JSD-informed density-matrix correction

`jsd_predictive_zne_density_matrix(rho_at_scales, noise_factors)` is `zne_density_matrix`
for exactly 3 equally-spaced scales, with a further nudge based on how much the
noise-scale-to-output-distribution relationship deviates from smooth (measured via
Jensen-Shannon divergence between consecutive scales' measurement distributions,
positive-only — the nudge is rectified to 0 whenever the signal would predict a
different direction, since an earlier unrectified version helped only 5/16 test points
despite the signal itself correlating with success). Validated on 46 real activated
points (photon-loss noise, 6 independent seeds): improves fidelity on 35/46 (76.1%),
mean gain +0.0055. Needs no oracle access to an ideal state, unlike naively reusing
`calculate_delta_preemp` with an external signal (tried first, found negligible).

### `project_to_physical`

Projects a Hermitian, trace-1 matrix onto the nearest true density matrix (Hermitian,
trace 1, positive-semidefinite) in Frobenius distance — the same problem
[Smolin, Gambetta & Smith 2012](https://arxiv.org/abs/1106.5458) solve by iterative
eigenvalue clipping, solved here instead as Euclidean projection of the eigenvalues onto
the probability simplex: a fully vectorized, `jax.jit`-traceable algorithm for the same
convex optimization problem (unique global minimum, so any correct algorithm agrees).
Verified to machine precision (~1e-15) against the paper's own worked example.

### `uhlmann_fidelity` stays finite at degenerate eigenvalues

`uhlmann_fidelity` is differentiable through both arguments, including when `rho_A` has
(near-)degenerate eigenvalues — common for a near-pure state's noisy density matrix.
JAX's built-in `eigh` gradient divides by `lambda_i - lambda_j` and returns NaN there
(a known upstream limitation); `uhlmann_fidelity` uses a custom JVP rule internally that
masks that term to 0 for near-degenerate pairs instead, verified against finite
differences to exact agreement across non-degenerate, 2-fold, and 3-fold degenerate test
matrices.

### `jax.jit`-compiled entry points

Every function above except Step 7's `bounded_exponential_extrapolate` has a `_jit`
counterpart (`richardson_extrapolate_jit`, `zero_noise_extrapolation_jit`,
`polynomial_extrapolate_jit`, `uhlmann_fidelity_jit`, `zne_density_matrix_jit`) for
callers inside an already-jitted pipeline (e.g.
`jax.lax.scan`) who don't want a host round-trip per call. Each skips the eager
version's own dtype auto-detection and argument validation — callers pass already-cast
`complex128`/`float64` arrays themselves — and `polynomial_extrapolate_jit`/
`zne_density_matrix_jit` require `degree` as a static argument.

### Why `zne_density_matrix` defaults to `degree=2`, not exact interpolation

More noise-scale points make exact interpolation (`richardson_extrapolate`) *worse*
under real statistical noise — Lagrange coefficients grow with point count, so the fit
increasingly forces itself through every noisy sample exactly. Measured directly
(4 qubits, all 5 noise channels, 5 seeds, same total measurement budget): exact
interpolation's mean fidelity gain dropped from +0.148 (3 points) to +0.081 (5 points)
to -0.220 (5 closely-spaced points — actively worse than no correction). A degree-2
least-squares fit on the same extra points instead reduces variance (std 0.062 to
0.035-0.046) at comparable or better mean gain, which is why `zne_density_matrix` uses
`polynomial_extrapolate` rather than exact interpolation by default.

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
