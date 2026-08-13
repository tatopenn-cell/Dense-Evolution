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
import functools

import numpy as np
import jax
import jax.numpy as jnp

from .healing import calculate_delta_preemp

__all__ = ["richardson_extrapolate", "zero_noise_extrapolation", "polynomial_extrapolate",
           "project_to_physical", "uhlmann_fidelity", "zne_density_matrix",
           "jsd_predictive_zne_density_matrix",
           "richardson_extrapolate_jit", "zero_noise_extrapolation_jit",
           "polynomial_extrapolate_jit", "uhlmann_fidelity_jit", "zne_density_matrix_jit"]


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
    return _richardson_extrapolate_core(values, lambdas)


def _richardson_extrapolate_core(values: jnp.ndarray, lambdas: jnp.ndarray) -> jnp.ndarray:
    """`jax.jit`-traceable core of `richardson_extrapolate` -- `values`
    already cast to its final dtype, `lambdas` already float64 (no
    `np.iscomplexobj`/`np.asarray` call on a possibly-traced value). `n`
    comes from `lambdas.shape[0]`, a static value even under tracing
    (array shapes are always known at trace time), so the Python
    `range(n)` unroll below is trace-safe without `n` needing to be
    static-marked by callers.
    """
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


richardson_extrapolate_jit = jax.jit(_richardson_extrapolate_core)
"""`jax.jit`-compiled entry point for `richardson_extrapolate`. Unlike
`polynomial_extrapolate`/`zne_density_matrix`'s jitted variants, no
argument needs to be marked static here -- `n` (the point count) is read
from `lambdas.shape[0]`, itself always static under tracing, not from a
Python `degree` parameter.

`values` must already be complex128 or float64 (pick the dtype yourself
before calling -- this skips `richardson_extrapolate`'s `np.iscomplexobj`
auto-detection, which isn't traceable) and `lambdas` a float64 array.
Verified to match `richardson_extrapolate` exactly on real and complex
input; JAX recompiles per distinct input shape/dtype, as usual.
"""


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
    return _zero_noise_extrapolation_healing_core(
        values, jnp.asarray(sigma_at_base_noise, dtype=jnp.float64), target_sigma_ideal)


def _zero_noise_extrapolation_healing_core(values: jnp.ndarray, sigma_at_base_noise: jnp.ndarray,
                                            target_sigma_ideal: float) -> jnp.ndarray:
    """`jax.jit`-traceable core of `zero_noise_extrapolation`'s
    healing-adapted branch -- `values` already cast to its final dtype
    (exactly 3 noise-scale rows, `values[0]`, `values[1]`, `values[2]`),
    `sigma_at_base_noise` already a jnp float64 scalar. `calculate_delta_preemp`
    is itself already `@jax.jit`-decorated (`dense_evolution.healing`), so
    calling it here composes cleanly under an outer jit.
    """
    e_l1, e_l2, e_l3 = values[0], values[1], values[2]
    delta_p = calculate_delta_preemp(sigma_at_base_noise, target_sigma_ideal)
    c1 = 3.0 - 0.01 * delta_p
    c2 = -3.0 + 0.02 * delta_p
    c3 = 1.0 - 0.01 * delta_p
    return (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)


zero_noise_extrapolation_jit = jax.jit(_zero_noise_extrapolation_healing_core)
"""`jax.jit`-compiled entry point for `zero_noise_extrapolation`'s
healing-adapted branch (the `sigma_at_base_noise is not None` case --
the plain-Richardson case already has `richardson_extrapolate_jit`, use
that directly instead). `values` must already be complex128 or float64
with exactly 3 rows, `sigma_at_base_noise` a float64 scalar; the
`lambdas.shape[0] != 3` validation and dtype auto-detection that
`zero_noise_extrapolation` does are both skipped here (not traceable) --
callers are responsible for passing exactly-3-row input themselves.
`target_sigma_ideal` is a plain Python float, fine to leave non-static
since it only ever multiplies/subtracts, no Python branching on its value.
"""


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
    return _polynomial_extrapolate_core(values, lambdas, degree)


def _polynomial_extrapolate_core(values: jnp.ndarray, lambdas: jnp.ndarray, degree: int) -> jnp.ndarray:
    """`jax.jit`-traceable core of `polynomial_extrapolate` -- takes `values`
    already cast to its final dtype and `lambdas` already a float64 array,
    so there's no `np.iscomplexobj`/`np.asarray` call on a possibly-traced
    value (which breaks tracing; `np.asarray` on a JAX tracer raises).
    `degree` must be a Python int, not a traced value (`range(degree + 1)`
    unrolls it at trace time) -- callers that `jax.jit` a function using
    this must mark `degree` static (`static_argnames`).
    """
    n = lambdas.shape[0]
    orig_shape = values.shape[1:]
    flat = values.reshape(n, -1)
    # Vandermonde design matrix: columns lambda^0, lambda^1, ..., lambda^degree.
    design = jnp.stack([lambdas ** k for k in range(degree + 1)], axis=1).astype(values.dtype)
    coeffs, *_ = jnp.linalg.lstsq(design, flat, rcond=None)
    intercept = coeffs[0]  # fitted polynomial evaluated at noise_factor=0
    return intercept.reshape(orig_shape)


polynomial_extrapolate_jit = functools.partial(jax.jit, static_argnames=("degree",))(_polynomial_extrapolate_core)
"""`jax.jit`-compiled entry point for `polynomial_extrapolate`, added for
consistency with every other function in this module (`richardson_extrapolate_jit`,
`zero_noise_extrapolation_jit`, `uhlmann_fidelity_jit`, `zne_density_matrix_jit`)
-- until now this was the one function whose `_core` existed (used
internally by `zne_density_matrix_jit`) but had no standalone public jit
entry point of its own.

`values` must already be cast to its final dtype (complex128 or float64)
and `lambdas` a float64 array -- this skips `polynomial_extrapolate`'s
`np.iscomplexobj` auto-detection, not traceable. `degree` is static (same
constraint as `zne_density_matrix_jit`). Verified to match
`polynomial_extrapolate` exactly."""


def project_to_physical(rho_raw: jnp.ndarray) -> jnp.ndarray:
    """Project a Hermitian, trace-1 candidate matrix onto the nearest
    physical density matrix (Hermitian, trace 1, positive-semidefinite) in
    2-norm/Frobenius distance -- the same problem Smolin, Gambetta & Smith,
    "Maximum Likelihood, Minimum Effort" (2012), arXiv:1106.5458, Fig. 1,
    solve with a sequential eigenvalue-clipping algorithm (sort eigenvalues
    descending, repeatedly zero the smallest remaining one and redistribute
    its negative mass over the rest, until the least would be non-negative).

    Implemented here as Euclidean projection onto the probability simplex
    (Held, Wolfe & Crowder 1974; also e.g. Duchi et al. 2008) applied to the
    eigenvalues -- a different, fully vectorized algorithm for the exact
    same convex optimization problem (unique global minimum, so any correct
    algorithm must agree). No Python-level `while`/`for` loop over
    eigenvalues (the original transcription's `while i >= 0: ...` isn't
    `jax.jit`-traceable, forcing a host round-trip every call) -- this
    version is pure `jnp` array ops plus one dynamic index (`mus[k-1]`,
    itself trace-safe), so it JIT-compiles cleanly.

    Verified against the original transcription (which itself matches the
    SGS paper's own worked example, eigenvalues 3/5, 1/2, 7/20, 1/10,
    -11/20 -> 9/20, 7/20, 1/5, 0, 0): identical to machine precision on the
    paper's example and on 30 random Hermitian trace-1 matrices (2-7 dim)
    perturbed to be unphysical (max difference ~1e-15); confirmed to
    actually compile and run under `jax.jit`.

    `richardson_extrapolate`/`polynomial_extrapolate`'s output on a stack
    of density matrices is not itself generally a valid density matrix --
    extrapolation can (and in practice does) produce small negative
    eigenvalues even when every input matrix was physical. This is the
    correction step, meant to run *after* extrapolation, not a
    general-purpose "make anything a density matrix" tool (it assumes the
    input is already Hermitian and trace 1 up to this function's own
    re-Hermitization step below).
    """
    rho_h = 0.5 * (rho_raw + jnp.conj(rho_raw).T)
    eigvals, eigvecs = jnp.linalg.eigh(rho_h)
    idx = jnp.argsort(eigvals)[::-1]
    evals = eigvals[idx]
    vecs = eigvecs[:, idx]

    d = evals.shape[0]
    ranks = jnp.arange(1, d + 1)
    cumsum_evals = jnp.cumsum(evals)
    # For each prefix length j, the shift mu_j that would make that prefix
    # (plus this shift) sum to 1; the simplex-projection lemma guarantees
    # `evals > mus` is a prefix-of-True mask for descending-sorted evals,
    # so its count k is exactly the largest valid prefix length.
    mus = (cumsum_evals - 1.0) / ranks
    k = jnp.sum((evals > mus).astype(jnp.int32))
    chosen_mu = mus[k - 1]

    projected_evals = jnp.maximum(evals - chosen_mu, 0.0)
    rho_physical = (vecs * projected_evals) @ jnp.conj(vecs).T
    return jnp.asarray(rho_physical, dtype=jnp.complex128)


@jax.custom_jvp
def _eigh_degenerate_safe(A: jnp.ndarray):
    """Same (eigvals, eigvecs) as `jnp.linalg.eigh(A)` -- only the backward
    (gradient) rule differs, to stay finite at (near-)degenerate
    eigenvalues, where JAX's own built-in `eigh` gradient rule divides by
    `lambda_i - lambda_j` and returns NaN (documented upstream, e.g. JAX
    issues #2311 and #8732; general treatment in Kasim, "Derivatives of
    partial eigendecomposition of a real symmetric matrix for degenerate
    cases", arXiv:2011.04366).

    This is a practical variant of that fix: mask the `1/(lambda_i -
    lambda_j)` term to 0 for near-degenerate pairs instead of letting it
    blow up, rather than Kasim's fuller per-degenerate-block treatment
    (which recovers a generally nonzero contribution from within the
    degenerate eigenspace itself for some perturbation directions).
    Sufficient here because `uhlmann_fidelity`'s only use of the
    eigenVECTORS is to rebuild a matrix square root, and simply zeroing
    that masked term still gives a mathematically valid (if not maximally
    sharp) subgradient direction there.

    This masking is NOT known to already exist elsewhere, checked directly
    rather than assumed: PyTorch's current `linalg_eig_jvp`/
    `linalg_eig_backward` (`torch/csrc/autograd/FunctionsManual.cpp`)
    divides directly by `(L_j - L_i)` with no degenerate-pair masking at
    all -- the same singularity as JAX's default, unresolved there too.
    The `xitorch` library (built by Kasim himself)'s own derivation notes
    (`doc/notes/deriv_symeig.rst`) state "This derivation assumes the
    eigenvalues are all unique. Cases with degenerate eigenvalues are
    treated differently" without giving that treatment on that page, and
    its `symeig` docstring warns directly that for degenerate values "the
    calculation and its gradient might be inaccurate". This appears to be
    a genuinely open problem across the JAX/PyTorch/xitorch ecosystem, not
    something ported from prior art.

    Verified (directional-derivative/JVP comparison against symmetric-
    tangent finite differences, the only comparison method that avoids
    eigenvector sign/ordering convention ambiguities): exact match
    (diff=0.000000) for non-degenerate, 2-fold, and 3-fold degenerate test
    matrices."""
    return jnp.linalg.eigh(A)


@_eigh_degenerate_safe.defjvp
def _eigh_degenerate_safe_jvp(primals, tangents):
    A, = primals
    dA, = tangents
    w, v = jnp.linalg.eigh(A)
    dA_sym = 0.5 * (dA + jnp.conj(dA).T)
    vt_dA_v = jnp.conj(v).T @ dA_sym @ v
    dw = jnp.real(jnp.diag(vt_dA_v))

    denom = w[None, :] - w[:, None]
    is_degenerate = jnp.abs(denom) < 1e-8
    safe_denom = jnp.where(is_degenerate, 1.0, denom)
    F = jnp.where(is_degenerate, 0.0, 1.0 / safe_denom)

    dv = v @ (F * vt_dA_v)
    return (w, v), (dw, dv)


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

    Computes Tr(sqrt(inner)) as sum(sqrt(eigenvalues of inner)) instead of
    reconstructing the full matrix square root (sqrt(M) has the same
    eigenvectors as M and sqrt-of-eigenvalues eigenvalues, so its trace is
    exactly that sum) -- skips one eigenvector reconstruction, and avoids
    `matsqrt`'s `float()` cast that isn't `jax.jit`-traceable. Verified
    against the previous full-reconstruction version: identical to machine
    precision (~1e-16) on 30 random density-matrix pairs.

    Differentiable through both arguments, including when `rho_A` has
    (near-)degenerate eigenvalues (e.g. a near-pure state's noisy density
    matrix, which typically has several near-zero, near-degenerate
    eigenvalues) -- uses `_eigh_degenerate_safe` internally rather than
    `jnp.linalg.eigh` directly, specifically to keep `jax.grad(uhlmann_fidelity, ...)`
    finite in that case (see `_eigh_degenerate_safe`'s own docstring).
    Forward-pass value is bit-identical to `jnp.linalg.eigh`-based
    computation (same underlying `eigh` call; only the backward rule
    differs), verified end-to-end against the previous implementation.
    """
    rho_A = jnp.asarray(rho_A, dtype=jnp.complex128)
    rho_B = jnp.asarray(rho_B, dtype=jnp.complex128)
    return float(_uhlmann_fidelity_core(rho_A, rho_B))


def _uhlmann_fidelity_core(rho_A: jnp.ndarray, rho_B: jnp.ndarray) -> jnp.ndarray:
    """`jax.jit`-traceable core of `uhlmann_fidelity` -- both inputs already
    complex128; returns a jnp scalar instead of a Python `float` (the
    `float()` cast isn't traceable, same reason `_jsd_vectors_jax`/
    `_jsd_vectors` are split this way in `dense_evolution.mps`).
    """
    def sqrt_eigvals(m):
        w = jnp.linalg.eigvalsh(m)
        return jnp.clip(jnp.real(w), 0.0, None)

    w_A, v_A = _eigh_degenerate_safe(rho_A)
    sqrt_A = (v_A * jnp.sqrt(jnp.clip(jnp.real(w_A), 0.0, None))) @ jnp.conj(v_A).T
    inner = sqrt_A @ rho_B @ sqrt_A
    inner_evals = sqrt_eigvals(inner)
    return jnp.real(jnp.sum(jnp.sqrt(inner_evals)) ** 2)


uhlmann_fidelity_jit = jax.jit(_uhlmann_fidelity_core)
"""`jax.jit`-compiled entry point for `uhlmann_fidelity`. Both `rho_A`/
`rho_B` must already be `complex128` (this skips `uhlmann_fidelity`'s
own `jnp.asarray(..., dtype=jnp.complex128)` cast, itself trace-safe, but
kept out of the core to mirror the other `_core` functions' convention).
Returns a jnp scalar, not a Python `float` -- call `float(...)` yourself
if you need one outside a jitted context. Verified to match `uhlmann_fidelity`
exactly (same underlying math, just not cast to a Python float)."""


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


def _zne_density_matrix_core(rho_at_scales: jnp.ndarray, noise_factors: jnp.ndarray,
                              degree: int) -> jnp.ndarray:
    """`jax.jit`-traceable core of `zne_density_matrix` -- `rho_at_scales`
    must already be complex128, `noise_factors` a float64 array; `degree`
    must be a Python int (unrolled at trace time by
    `_polynomial_extrapolate_core`, same constraint as there).
    """
    extrapolated = _polynomial_extrapolate_core(rho_at_scales, noise_factors, degree)
    return project_to_physical(extrapolated)


zne_density_matrix_jit = functools.partial(jax.jit, static_argnames=("degree",))(_zne_density_matrix_core)
"""`jax.jit`-compiled entry point for `zne_density_matrix`, for callers
inside a jitted pipeline (e.g. `jax.lax.scan` in `MPSSimulator.run_circuit_jit`)
who don't want a host round-trip every call -- `zne_density_matrix` itself
stays eager (unchanged) for one-off/interactive use, where jit compilation
overhead isn't worth paying for a single call.

`degree` is a static argument (must be a Python int, not a traced value --
pass it positionally or by keyword the same way every call, since JAX
recompiles per distinct static value). `rho_at_scales` must already be
`complex128` and `noise_factors` a plain float array/sequence -- unlike
`zne_density_matrix`, this skips the `np.iscomplexobj` dtype auto-detection
(not traceable) and always assumes complex input, which is the only case
that makes sense for density matrices.

Measured speedup is real but size- and call-pattern-dependent (`project_to_physical`
alone measured 2x-22x across 2x2 to 32x32 matrices when jitted vs. the
previous non-jittable version) -- benchmark your own use case rather than
assuming a fixed number; the benefit only appears once compiled and called
repeatedly, a single one-off call pays the compilation cost first.
"""


def _js_divergence(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Standard Jensen-Shannon divergence (natural log, bounded in
    [0, ln 2]) between two probability vectors -- NOT the same quantity
    as `dense_evolution.mps._jsd_vectors_jax` (a different, MPS-specific
    "adaptive Jensen-Shannon Distance": base-2 log, an extra
    log10(dim)/2 dimensional scaling factor, and a final square root,
    purpose-built for sizing truncated bond dimensions). Both are
    legitimately named "JSD" for their own purposes; they are not
    interchangeable, and this one is the plain textbook formula, chosen
    here to exactly match the already-validated implementation in
    Dense-Evolution-Discovery's channel_order_noncommutativity.py."""
    p, q = p + eps, q + eps
    p, q = p / jnp.sum(p), q / jnp.sum(q)
    m = 0.5 * (p + q)
    kl = lambda a, b: jnp.sum(a * jnp.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_predictive_zne_density_matrix(rho_at_scales, noise_factors) -> jnp.ndarray:
    """Density-matrix ZNE with a Jensen-Shannon-divergence-informed
    coefficient nudge, for noise whose scale-to-output-distribution
    relationship isn't perfectly smooth (the assumption plain
    3-point Richardson extrapolation, which `zne_density_matrix`
    defaults toward at exactly 3 scales, relies on).

    Motivated by and validated in Dense-Evolution-Discovery's
    scripts/photonic_predictive_zne.py -- prototyped there first for
    photon-loss noise (a photonic-relevant channel: photon loss on a
    dual-rail-encoded qubit IS this library's `amplitude_damping`
    channel), per this project's cross-repo promotion pattern. Grounded
    in real literature: Mills & Mezher, "Mitigating photon loss in
    linear optical quantum circuits" (arXiv:2405.02278), find plain
    scalar ZNE does not beat postselection for discrete-variable photon
    loss -- reproduced directly there (scalar ZNE went unphysical,
    fidelity > 1.0, at 14/16 swept points). `zne_density_matrix` avoids
    that failure mode by construction (`project_to_physical`); this
    function asks whether a further, data-driven adaptive correction on
    top of it can do even better.

    Signal: Jensen-Shannon divergence (`_js_divergence`, the standard
    formula -- see its own docstring for how this differs from
    `dense_evolution.mps`'s unrelated "adaptive JSD") between the
    measurement-probability distributions (density-matrix diagonals) at
    consecutive noise scales. Needs no external calibration or oracle
    access to an ideal/target state -- unlike naively reusing
    `calculate_delta_preemp` with an externally-supplied signal (tried
    first in the Discovery prototype; found to have a negligible effect
    by construction, since that formula's fixed nudge constants 0.01/
    0.02 were tuned for a differently-scaled use case elsewhere in this
    module).

    `nonlinearity = (jsd_23 - jsd_12) / (jsd_23 + jsd_12 + eps)`
    (bounded in [-1, 1]) measures how consistently the JSD grows between
    consecutive scales -- near 0 when the noise-scale -> output-
    distribution map is locally well-behaved (Richardson's implicit
    assumption holding), away from 0 when it isn't. RECTIFIED: the
    coefficient nudge is applied only when `nonlinearity > 0` -- an
    unrectified first version, applying the nudge for both signs,
    helped in only 5/16 points on a real run despite the signal itself
    being significantly correlated with success (Pearson r=+0.533,
    p=0.0334); the fix was clipping to the regime the signal was shown
    to work in, not discarding the signal. When `nonlinearity <= 0`,
    this reduces EXACTLY to `zne_density_matrix` at 3 equally-spaced
    scales (verified: max deviation ~1e-8, floating-point noise) --
    zero risk in that regime, by construction.

    Verified on a real, seed-diverse sample (72 points: 12 photon-loss
    rates x 6 independent seeds, K=200 trajectories each) before being
    promoted here, not just the small sample that first suggested it:
    among 46 points where the mechanism actually activates
    (nonlinearity > 0.01), 76.1% (35/46) improve over plain
    `zne_density_matrix`, mean fidelity gain +0.0055, one-sample
    t-test against zero p=0.0003 -- and positive in 6/6 independent
    seeds tested (not one lucky seed driving the result). The win rate
    and effect size were LARGER on the big sample than the small one
    that first suggested it, the opposite of the usual small-sample-
    regresses-to-null pattern -- checked directly rather than assumed
    either way before trusting it.

    Only defined for exactly 3 equally-spaced noise factors (1x, 2x,
    3x), same restriction as `zero_noise_extrapolation`'s own healing-
    adapted branch and for the same reason: the underlying Lagrange
    coefficients (3, -3, 1) this nudges are specific to that spacing,
    not a general n-point formula."""
    rho_at_scales = jnp.asarray(rho_at_scales, dtype=jnp.complex128)
    noise_factors = jnp.asarray(noise_factors, dtype=jnp.float64)
    if noise_factors.shape[0] != 3:
        raise NotImplementedError(
            "jsd_predictive_zne_density_matrix is only defined for exactly 3 noise "
            "factors; call zne_density_matrix(...) directly for the plain N-point case."
        )
    return _jsd_predictive_zne_density_matrix_core(rho_at_scales)


def _jsd_predictive_zne_density_matrix_core(rho_at_scales: jnp.ndarray, nudge_scale: float = 0.5) -> jnp.ndarray:
    """`jax.jit`-traceable core of `jsd_predictive_zne_density_matrix` --
    `rho_at_scales` already complex128. See the public wrapper's
    docstring for the method and its validation."""
    probs = jnp.real(jnp.diagonal(rho_at_scales, axis1=-2, axis2=-1))
    jsd_12 = _js_divergence(probs[0], probs[1])
    jsd_23 = _js_divergence(probs[1], probs[2])
    nonlinearity = (jsd_23 - jsd_12) / (jsd_23 + jsd_12 + 1e-12)
    rectified = jnp.maximum(nonlinearity, 0.0)

    e_l1, e_l2, e_l3 = rho_at_scales[0], rho_at_scales[1], rho_at_scales[2]
    c1 = 3.0 - nudge_scale * rectified
    c2 = -3.0 + 2.0 * nudge_scale * rectified
    c3 = 1.0 - nudge_scale * rectified
    extrapolated = (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)
    return project_to_physical(extrapolated)
