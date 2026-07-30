"""
Zero-Noise Extrapolation (ZNE)
-------------------------------
Standard error-mitigation entry points, named the way the field already
names them (Richardson extrapolation, noise factors, zero-noise
extrapolation -- same vocabulary as e.g. Mitiq's `zne` API), so callers
and tooling can find "ZNE" without first learning Dense-Evolution's
internal healing vocabulary.

This module composes `dense_evolution.healing`'s existing primitives
(`calculate_delta_preemp`, ...) -- it does not rename or replace them.
"""
import numpy as np
import jax.numpy as jnp

from .healing import calculate_delta_preemp

__all__ = ["richardson_extrapolate", "zero_noise_extrapolation", "polynomial_extrapolate",
           "project_to_physical", "uhlmann_fidelity", "zne_density_matrix"]


def richardson_extrapolate(expectation_values, noise_factors) -> jnp.ndarray:
    """Polynomial (Lagrange) Richardson extrapolation to zero noise.

    `expectation_values[i]` is the value measured/simulated at noise scale
    `noise_factors[i]` (e.g. 1x, 2x, 3x folded/scaled noise) -- a scalar,
    or itself an array (e.g. a full probability distribution sampled at
    that noise scale; extrapolated elementwise). Returns the extrapolated
    zero-noise estimate, same shape as one `expectation_values[i]`. Works
    for any number of points and any (not necessarily equally spaced)
    noise factors; for the common 3-point case at noise_factors=(1,2,3)
    this reduces exactly to the textbook coefficients (3, -3, 1).

    `expectation_values` may be complex (e.g. density matrix entries,
    which are complex off-diagonal in general) -- dtype is picked from
    the input itself (complex128 if complex, float64 otherwise, matching
    this function's previous always-float64 behavior for real input
    exactly). Found via a real test case: forcing float64 unconditionally
    silently discarded the imaginary part of complex input with no
    visible error, only a low-signal ComplexWarning easy to miss --
    confirmed directly (`richardson_extrapolate([1+2j, 3+4j], ...)` used
    to return a purely real result, dropping real information).
    """
    lambdas = jnp.asarray(noise_factors, dtype=jnp.float64)
    values_dtype = jnp.complex128 if np.iscomplexobj(np.asarray(expectation_values)) else jnp.float64
    values = jnp.asarray(expectation_values, dtype=values_dtype)
    n = lambdas.shape[0]

    def lagrange_coeff(i):
        others = jnp.concatenate([lambdas[:i], lambdas[i + 1:]])
        return jnp.prod((0.0 - others) / (lambdas[i] - others))

    coeffs = jnp.stack([lagrange_coeff(i) for i in range(n)])
    # Broadcast coeffs against the LEADING axis of values (the "one row per
    # noise scale" axis), not jnp's default trailing-axis alignment --
    # values may itself be array-valued per scale (values.shape = (n,
    # *extra_dims)), e.g. a whole probability distribution rather than a
    # bare scalar. A no-op reshape when values is 1-D (the scalar case),
    # so existing scalar callers are unaffected.
    coeffs = coeffs.reshape((n,) + (1,) * (values.ndim - 1))
    return jnp.sum(coeffs * values, axis=0)


def zero_noise_extrapolation(expectation_values, noise_factors,
                              sigma_at_base_noise=None,
                              target_sigma_ideal: float = 10.0) -> jnp.ndarray:
    """Zero-Noise Extrapolation -- plain, or healing-adapted when a
    coherence signal is available.

    Without `sigma_at_base_noise`: standard Richardson ZNE
    (`richardson_extrapolate`).

    With `sigma_at_base_noise` (the measured/simulated coherence sigma at
    the base, unscaled noise level): the 3 Richardson coefficients are
    perturbed by `dense_evolution.healing.calculate_delta_preemp` -- the
    normalized deviation between the observed sigma and the ideal target
    -- then renormalized to sum to 1. This is Dense-Evolution's
    "predictive healing" ZNE variant: when the observed coherence is off
    the ideal target, the extrapolation is nudged accordingly instead of
    trusting the 3 raw noise-scaled points equally.

    The healing-adapted path currently only supports exactly 3 noise
    factors (the case it has been derived and tested against); passing
    `sigma_at_base_noise` with any other point count raises
    NotImplementedError rather than silently generalizing an unverified
    formula.
    """
    if sigma_at_base_noise is None:
        return richardson_extrapolate(expectation_values, noise_factors)

    lambdas = jnp.asarray(noise_factors, dtype=jnp.float64)
    if lambdas.shape[0] != 3:
        raise NotImplementedError(
            "Healing-adapted ZNE is only defined for exactly 3 noise factors; "
            "call richardson_extrapolate(...) directly for the plain N-point case."
        )
    values_dtype = jnp.complex128 if np.iscomplexobj(np.asarray(expectation_values)) else jnp.float64
    values = jnp.asarray(expectation_values, dtype=values_dtype)
    e_l1, e_l2, e_l3 = values[0], values[1], values[2]

    delta_p = calculate_delta_preemp(jnp.asarray(sigma_at_base_noise, dtype=jnp.float64),
                                      target_sigma_ideal)
    c1 = 3.0 - 0.01 * delta_p
    c2 = -3.0 + 0.02 * delta_p
    c3 = 1.0 - 0.01 * delta_p
    return (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)


def polynomial_extrapolate(expectation_values, noise_factors, degree: int = 2) -> jnp.ndarray:
    """Least-squares polynomial extrapolation to zero noise.

    Generalizes `richardson_extrapolate`: fits a degree-`degree` polynomial
    to `(noise_factors, expectation_values)` by ordinary least squares and
    evaluates it at zero. With exactly `degree + 1` points the fit is the
    unique interpolating polynomial, mathematically identical to
    `richardson_extrapolate` at that point count (verified directly, both
    for real and complex input). With MORE than `degree + 1` points it
    becomes an overdetermined fit -- the extra points average down
    statistical noise instead of forcing the polynomial through every
    noisy sample exactly, trading a small amount of interpolation bias
    for reduced variance.

    This matters in practice, not just in theory: adding more noise-scale
    points to *exact* interpolation (`richardson_extrapolate`) makes
    extrapolation WORSE under real statistical noise, because Lagrange
    coefficients grow with point count (worse still with closely-spaced
    points -- a Runge's-phenomenon-like effect). Measured directly on the
    density-matrix healing experiment (`experiments/matrix_healing_zne_sweep.py`'s
    setup, n=4 qubits, all 5 noise channels, 5 seeds): exact interpolation's
    mean fidelity-delta dropped from +0.148 (3 points) to +0.081 (5 points,
    same spacing) to -0.220 (5 points, denser spacing -- actively worse
    than not correcting). A degree-2 least-squares fit fed the same extra
    points instead REDUCES variance (std 0.062 -> 0.035-0.046) at
    comparable or better mean delta, because the extra points are no
    longer forced to satisfy an increasingly ill-conditioned exact fit.
    This is why `zne_density_matrix` uses this function (degree=2) instead
    of `richardson_extrapolate` by default.

    Raises ValueError if fewer than `degree + 1` points are given (the fit
    would be underdetermined).
    """
    lambdas = jnp.asarray(noise_factors, dtype=jnp.float64)
    n = lambdas.shape[0]
    if n < degree + 1:
        raise ValueError(
            f"polynomial_extrapolate needs at least degree+1={degree + 1} noise "
            f"factors for a degree-{degree} fit, got {n}."
        )
    values_dtype = jnp.complex128 if np.iscomplexobj(np.asarray(expectation_values)) else jnp.float64
    values = jnp.asarray(expectation_values, dtype=values_dtype)
    orig_shape = values.shape[1:]
    flat = values.reshape(n, -1)

    # Vandermonde design matrix: columns lambda^0, lambda^1, ..., lambda^degree.
    design = jnp.stack([lambdas ** k for k in range(degree + 1)], axis=1).astype(values_dtype)
    coeffs, *_ = jnp.linalg.lstsq(design, flat, rcond=None)
    intercept = coeffs[0]  # fitted polynomial evaluated at noise_factor=0
    return intercept.reshape(orig_shape)


def project_to_physical(rho_raw: jnp.ndarray) -> jnp.ndarray:
    """Project a Hermitian, trace-1 candidate matrix onto the nearest
    physical density matrix (Hermitian, trace 1, positive-semidefinite) in
    2-norm/Frobenius distance.

    Implements the "Fast algorithm for Subproblem 1" of Smolin, Gambetta &
    Smith, "Maximum Likelihood, Minimum Effort" (2012), arXiv:1106.5458,
    Fig. 1: sort eigenvalues descending, then repeatedly zero out the
    smallest remaining eigenvalue and redistribute its (negative) mass
    equally over the eigenvalues not yet zeroed, until the least of those
    would be non-negative. Transcribed and checked against the paper's own
    worked numeric example (eigenvalues 3/5, 1/2, 7/20, 1/10, -11/20 ->
    9/20, 7/20, 1/5, 0, 0) before use.

    `richardson_extrapolate`'s output on a stack of density matrices is not
    itself generally a valid density matrix -- polynomial extrapolation can
    (and in practice does) produce small negative eigenvalues even when
    every input matrix was physical. This is the correction step, meant to
    run *after* extrapolation, not a general-purpose "make anything a
    density matrix" tool (it assumes the input is already Hermitian and
    trace 1 up to the projection's own re-Hermitization step below).
    """
    rho_h = 0.5 * (rho_raw + jnp.conj(rho_raw).T)
    eigvals, eigvecs = jnp.linalg.eigh(rho_h)
    # jnp.linalg.eigh returns ascending order; the algorithm as stated
    # works on eigenvalues sorted descending -- reverse both the values and
    # the matching eigenvector columns.
    eigvals = np.asarray(eigvals)[::-1]
    eigvecs_np = np.asarray(eigvecs)[:, ::-1]

    d = len(eigvals)
    lam = eigvals.copy()
    a = 0.0
    i = d - 1
    while i >= 0:
        if lam[i] + a / (i + 1) >= 0:
            break
        a += lam[i]
        lam[i] = 0.0
        i -= 1
    for j in range(i + 1):
        lam[j] = lam[j] + a / (i + 1)

    rho_physical = (eigvecs_np * lam) @ eigvecs_np.conj().T
    return jnp.asarray(rho_physical, dtype=jnp.complex128)


def uhlmann_fidelity(rho_A: jnp.ndarray, rho_B: jnp.ndarray) -> float:
    """Uhlmann fidelity F(rho_A, rho_B) = (Tr sqrt(sqrt(rho_A) rho_B sqrt(rho_A)))^2.

    Reduces to |<psi_A|psi_B>|^2 when both inputs are pure-state density
    matrices (verified directly). Validation-only: this is meant to grade
    a correction against a known ideal state, never to feed into one --
    passing it a target/ideal density matrix as an input to
    `zne_density_matrix` or any extrapolation step would be using held-out
    ground truth to guide the algorithm (oracle access), not a legitimate
    error-mitigation technique. Keeping ideal-state comparison to this
    function only, rather than plumbing it into the correction functions
    at all, makes that boundary structural rather than a convention callers
    have to remember.
    """
    def matsqrt(m):
        w, v = jnp.linalg.eigh(m)
        w = jnp.clip(jnp.real(w), 0.0, None)
        return (v * jnp.sqrt(w)) @ jnp.conj(v).T

    rho_A = jnp.asarray(rho_A, dtype=jnp.complex128)
    rho_B = jnp.asarray(rho_B, dtype=jnp.complex128)
    sqrt_A = matsqrt(rho_A)
    inner = matsqrt(sqrt_A @ rho_B @ sqrt_A)
    return float(jnp.real(jnp.trace(inner)) ** 2)


def zne_density_matrix(rho_at_scales, noise_factors, degree: int = 2) -> jnp.ndarray:
    """Zero-Noise Extrapolation for density matrices.

    `rho_at_scales[i]` is a noisy density-matrix estimate (e.g. from a
    Monte-Carlo/shot ensemble) at noise scale `noise_factors[i]`.
    Extrapolates to zero noise via `polynomial_extrapolate` (least-squares,
    complex-safe, degree=2 by default) and projects the result onto the
    nearest physical density matrix (`project_to_physical`), since the raw
    extrapolated matrix is not generally positive-semidefinite even when
    every input was.

    Uses `polynomial_extrapolate` rather than exact `richardson_extrapolate`
    because, with exactly 3 noise scales (the original design point), the
    two are mathematically identical -- but `polynomial_extrapolate` stays
    well-behaved (reduced variance) when a caller passes MORE than 3 scales,
    where exact interpolation instead gets WORSE (see
    `polynomial_extrapolate`'s docstring for the measured numbers). This
    makes "pass more noise-scale points" a safe thing to try rather than a
    trap.

    Honest findings, both against a GHZ-state ideal target and
    `dense_evolution.registry.NoiseModel` noise at base_p=0.05, scales
    1x/2x/3x unless noted, `uhlmann_fidelity` against the true ideal state
    used only to grade the result -- never as input to any step above:

    - `experiments/matrix_healing_zne.py`: 2-qubit Bell state, depolarizing
      noise, K=200-trajectory estimate per scale, averaged over 4 seeds --
      raw fidelity ~0.865, corrected ~0.947 (+0.08), positive on every seed
      tested.
    - `experiments/matrix_healing_zne_sweep.py`: 2-5 qubits x all 5
      `NoiseModel` channels (depolarizing, bitflip, phaseflip,
      amplitude_damping, combined) x 5 seeds, K=400 trajectories per scale
      (100 runs total) -- **96/100 positive, mean delta +0.12**, and every
      single (qubit count, noise channel) combination is net positive on
      average, growing to +0.20-0.25 at 5 qubits for depolarizing/bitflip.
      The 4 remaining negative runs are small (worst -0.02) and consistent
      with residual Monte Carlo noise, not a systematic failure mode.
      (These specific numbers were measured with exact 3-point
      interpolation, which is identical to this function's degree=2
      default at 3 points -- unaffected by the switch.)
    - An earlier draft of this sweep (K=150, 3 seeds) had reported
      phaseflip/amplitude_damping as "unreliable" -- re-investigated rather
      than trusted, and confirmed to be a Monte Carlo undersampling
      artifact (extrapolation coefficients amplify input noise; an
      undersampled estimate makes the *corrected* result noisy even when
      the correction itself is sound), not a real limitation.
    - More noise-scale points, SAME total measurement budget (the fair
      comparison -- splitting a fixed number of trajectories across more
      points, not spending more): 3 points x K=400 (1200 total) vs. 5
      points x K=240 (1200 total) vs. 7 points x K=171 (~1200 total),
      n=4 qubits, all 5 noise channels, 5 seeds. 5 points matches or
      slightly beats the 3-point mean delta (+0.150 vs +0.148) with 19%
      lower variance (std 0.050 vs 0.062) -- a real, free improvement at
      the same experimental cost, not an artifact of spending more. 7
      points trades a little mean (+0.132) for still-lower variance (std
      0.043, 30% below baseline) -- a genuine tradeoff point, useful when
      reliability matters more than average performance. At exactly 3
      points this function is mathematically identical to
      `richardson_extrapolate` (verified to 1e-12) -- there is no free
      lunch there, the gain only appears once more points are used.
    - More noise-scale points, FIXED K per point instead (spending more
      total measurement, K=400 at every point count): with exact
      interpolation this makes things worse, not better (mean delta drops
      from +0.148 at 3 points to +0.081 at 5, and to -0.220 at 5
      closely-spaced points -- see `polynomial_extrapolate`'s docstring);
      with this function's degree=2 default it instead stays comparable
      in mean with lower variance (std 0.062 -> 0.035-0.046) -- confirms
      the safety property holds even when *not* holding budget fixed, on
      top of the fixed-budget gain above.

    Practical implication for callers regardless of `degree`: correction
    quality still depends on `rho_at_scales` being a reasonably low-noise
    estimate to begin with (large enough K, or equivalent) -- an
    extrapolation fit through pure noise cannot recover signal that isn't
    there. `degree` trades bias for variance: higher degree fits the true
    curve's shape more closely (less bias) but is more sensitive to
    per-point noise (more variance); degree=2 was chosen empirically as the
    best tested tradeoff, not a theoretical optimum for every regime.

    Do not pass a target/ideal density matrix into this function or use one
    to pick among candidate corrections -- see `uhlmann_fidelity`'s
    docstring for why.
    """
    extrapolated = polynomial_extrapolate(rho_at_scales, noise_factors, degree=degree)
    return project_to_physical(extrapolated)
