"""Classical-shadows-based estimator for `magic_entropy` (see
`magic_entropy.py` in this same subpackage) -- estimates the same
single-qubit non-stabilizerness quantity from randomized measurement
snapshots instead of requiring the exact density matrix.

Originated from a Colab proposal for a `dense_evolution/circuits/shadows.py`
module (following Huang, Kueng, Preskill 2020, "Predicting Many Properties
of a Quantum System from Very Few Measurements") with a real bug in its
purity estimator (a missing transpose in a U-statistic einsum contraction,
silent whenever every snapshot happened to be real-valued). Fixed, then
extended -- using the same multi-copy U-statistic trick the paper says
"readily generalizes to higher order polynomials" -- to estimate
`magic_entropy`'s reduced convolution matrix from shadow snapshots instead
of the exact rho. Matured across Dense-Evolution-Discovery Experiment 31
(three real gaps found and closed there before promotion: the purity
estimator bug, missing median-of-means robustness, and no
sample-complexity guidance):
https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_shadows_magic_entropy/

API SHAPE differs from every other function in this subpackage: sampling
(`sample_classical_shadow`) and estimation (`magic_entropy_from_shadows`)
are separate steps, matching how classical shadows work in general -- the
snapshot data can come from this simulator (`sample_classical_shadow` uses
oracle access to `rho`'s exact Born-rule probabilities, something only a
simulator has) or, in principle, from real hardware measurement outcomes
reconstructed the same way (`rho_hat = 3 U^dagger |b><b| U - I` per
snapshot, from a recorded basis+outcome).

Restricted to SINGLE-QUBIT density matrices, matching `magic_entropy`'s
own scope.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .magic_entropy import _KEY_UNITARY_K3

__all__ = [
    "sample_classical_shadow", "magic_entropy_from_shadows",
    "approx_shadow_std", "fit_shadow_sample_complexity",
]

# Single-qubit random-Pauli classical shadow protocol (Huang, Kueng,
# Preskill 2020, eq. 2-3): the three diagonalizing unitaries for X, Y, Z.
_H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)
_SDAG = jnp.array([[1.0, 0.0], [0.0, -1j]], dtype=jnp.complex128)
_BASIS_U = jnp.stack([_H, _H @ _SDAG, jnp.eye(2, dtype=jnp.complex128)])


def sample_classical_shadow(rho: jnp.ndarray, n_snapshots: int, seed: int = 0) -> jnp.ndarray:
    """Simulates the real single-qubit random-Pauli classical-shadow
    measurement protocol against a known `rho` (2x2, pure or mixed): for
    each of `n_snapshots` independent draws, picks a random Pauli basis
    uniformly, samples a computational-basis outcome from the true
    Born-rule probability under that basis (this simulator has oracle
    access to `rho`, unlike real hardware), then reconstructs the
    classical snapshot `rho_hat = 3 U^dagger |b><b| U - I`.

    Returns an `(n_snapshots, 2, 2)` complex128 array -- feed this
    directly into `magic_entropy_from_shadows`.

    Each individual `rho_hat` is NOT a valid density matrix on its own
    (can have negative eigenvalues) -- only the average over many
    snapshots converges to the true `rho`. Verified directly in
    Dense-Evolution-Discovery Experiment 31: the empirical mean over
    200,000 snapshots matched the true state to within 0.004.
    """
    rho = jnp.asarray(rho, dtype=jnp.complex128)
    key = jax.random.PRNGKey(seed)
    key_basis, key_bit = jax.random.split(key)
    bases = jax.random.randint(key_basis, (n_snapshots,), 0, 3)

    def prob0(basis_idx):
        u = _BASIS_U[basis_idx]
        rotated = u @ rho @ jnp.conj(u).T
        return jnp.clip(jnp.real(rotated[0, 0]), 0.0, 1.0)

    probs0 = jax.vmap(prob0)(bases)
    uniforms = jax.random.uniform(key_bit, (n_snapshots,))
    bits = (uniforms > probs0).astype(jnp.int32)

    def snapshot_matrix(basis_idx, bit):
        u = _BASIS_U[basis_idx]
        b_ket = jnp.array([1.0, 0.0], dtype=jnp.complex128) * (1 - bit) + \
            jnp.array([0.0, 1.0], dtype=jnp.complex128) * bit
        proj = jnp.outer(b_ket, jnp.conj(b_ket))
        return 3.0 * (jnp.conj(u).T @ proj @ u) - jnp.eye(2, dtype=jnp.complex128)

    return jax.vmap(snapshot_matrix)(bases, bits)


def _o_operators():
    """O_ab = V^dagger (|b><a| (x) I_4) V such that R_ab = Tr[O_ab . rho^{(x)3}]
    -- the same construction `magic_entropy.py`'s `_self_convolve_3_core`
    applies to exact `rho`, here turned into fixed operators so each entry
    is a LINEAR functional of `rho^{(x)3}` (the shape a shadow-snapshot
    U-statistic estimator needs). Reuses this package's own
    `_KEY_UNITARY_K3` directly rather than duplicating it (unlike
    Dense-Evolution-Discovery's per-script self-containment convention,
    this library's internal modules import from each other freely). The
    `|b><a|` (not `|a><b|`) projector is deliberate: a direct index-expansion
    check (not just the cyclic-trace derivation, which looks right on paper
    but hides the swap) showed `Tr[(|a><b| (x) I) M] = R_ba`, not `R_ab` --
    caught in Discovery Experiment 31 by a unit test checking matrix
    entries directly, not just the downstream entropy."""
    i4 = jnp.eye(4, dtype=jnp.complex128)
    v = _KEY_UNITARY_K3
    ops = {}
    for a in range(2):
        for b in range(2):
            proj_ba = jnp.zeros((2, 2), dtype=jnp.complex128).at[b, a].set(1.0)
            ops[(a, b)] = jnp.conj(v).T @ jnp.kron(proj_ba, i4) @ v
    return ops


_O_AB = _o_operators()


def _median_of_means(values: np.ndarray, n_groups: int) -> float:
    """Split real-valued `values` into `n_groups` contiguous batches,
    average each batch, then return the median of those batch means --
    Huang et al.'s standard robustification for shadow-estimator
    U-statistics. Unlike a single overall mean, the median tolerates up
    to `n_groups // 2` entirely corrupted/outlier batches (e.g. a
    systematic calibration fault affecting one contiguous stretch of a
    measurement run) without being dragged toward them -- verified
    directly in Dense-Evolution-Discovery Experiment 31: stays within 0.5
    of the true value at 40% of samples corrupted, while a plain mean is
    dragged from 1.0 to -19.4 under the same corruption."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    n_groups = max(1, min(n_groups, n))
    group_size = n // n_groups
    trimmed = values[: group_size * n_groups]
    group_means = trimmed.reshape(n_groups, group_size).mean(axis=1)
    return float(np.median(group_means))


def magic_entropy_from_shadows(shadow_snapshots: jnp.ndarray, n_groups: int = 20) -> float:
    """Estimates `magic_entropy(rho)` from classical shadow snapshots of
    `rho` (from `sample_classical_shadow`, or real hardware measurement
    data reconstructed the same way) instead of the exact density matrix.

    Groups the snapshots into disjoint triples, estimates each entry of
    the 3-copy self-convolution's reduced matrix `R` via median-of-means
    over `Tr[O_ab . (rho_hat_i (x) rho_hat_j (x) rho_hat_k)]` (unbiased,
    since each triple's three snapshots are independent unbiased
    estimators of `rho`; real and imaginary parts of each entry are
    median-of-means'd separately, the standard practical choice since a
    complex median has no single definition), then computes the von
    Neumann entropy of the (Hermitized, eigenvalue-clipped, trace-
    renormalized) ESTIMATED `R` classically -- entropy itself is never
    shadow-estimated directly, matching how Huang et al. handle their own
    Renyi-2 entanglement entropy example.

    Not `jax.jit`-compatible (unlike every other function in this
    subpackage): median-of-means uses `numpy.median`, which has no
    equivalent JAX primitive at this scale.

    See `approx_shadow_std`/`fit_shadow_sample_complexity` for how many
    snapshots this needs for a given error tolerance.
    """
    n = shadow_snapshots.shape[0]
    n_triples = n // 3
    if n_triples < 1:
        raise ValueError(f"need at least 3 shadow snapshots to form one triple, got {n}")
    triples = shadow_snapshots[: n_triples * 3].reshape(n_triples, 3, 2, 2)

    def triple_kron(t):
        return jnp.kron(jnp.kron(t[0], t[1]), t[2])

    rho3_batch = jax.vmap(triple_kron)(triples)

    r_hat = jnp.zeros((2, 2), dtype=jnp.complex128)
    for (a, b), o_ab in _O_AB.items():
        vals = np.array(jnp.einsum("ij,tji->t", o_ab, rho3_batch))
        real_part = _median_of_means(vals.real, n_groups)
        imag_part = _median_of_means(vals.imag, n_groups)
        r_hat = r_hat.at[a, b].set(real_part + 1j * imag_part)

    r_hat = 0.5 * (r_hat + jnp.conj(r_hat).T)  # enforce Hermiticity
    ev = jnp.linalg.eigvalsh(r_hat)
    safe_ev = jnp.clip(ev.real, 1e-9, None)
    safe_ev = safe_ev / jnp.sum(safe_ev)  # renormalize (estimation noise can shift trace off 1)
    return float(-jnp.sum(safe_ev * jnp.log2(safe_ev)))


# Empirically fitted in Dense-Evolution-Discovery Experiment 31 (20 trials
# per snapshot count on a |T> state, n_groups=20): std(n_snapshots) ~
# _FIT_C / n_snapshots ** _FIT_P. The T-state is a standard maximally-magic
# single-qubit state, a reasonable but NOT rigorously proven proxy for a
# "hard" case -- this is a rough built-in guide, not a formal guarantee for
# an arbitrary rho.
_FIT_C = 11.751
_FIT_P = 0.546


def approx_shadow_std(n_snapshots: int) -> float:
    """Rough approximate standard deviation (bits) of
    `magic_entropy_from_shadows`'s estimate at a given snapshot count,
    from an empirical fit (not a formal theorem) calibrated on a `|T>`
    state in Dense-Evolution-Discovery Experiment 31: 20 independent
    trials at each of 4 snapshot counts (3,000-100,000), a log-log linear
    regression gave `std(n) ~ 11.75 / n^0.546` -- the fitted exponent
    (0.546) is close to the ~0.5 ("error shrinks like 1/sqrt(n)") standard
    shadow/median-of-means theory predicts.

    This is a quick sanity-check fallback, not a guarantee for an
    arbitrary state -- call `fit_shadow_sample_complexity` on YOUR
    specific state if you need a real, state-calibrated error bound.
    """
    return _FIT_C / float(n_snapshots) ** _FIT_P


def fit_shadow_sample_complexity(rho: jnp.ndarray, exact_value: float, n_snapshots_list, n_trials: int, seed_base: int = 0):
    """Empirically measures `magic_entropy_from_shadows`'s standard
    deviation across `n_trials` independent shadow samplings at each
    snapshot count in `n_snapshots_list`, for the SPECIFIC state `rho`
    (rather than trusting `approx_shadow_std`'s built-in T-state-derived
    fallback), then fits `std(n) ~ C / n^p` via log-log linear regression
    -- the same method used to derive `approx_shadow_std`'s constants in
    the first place (Dense-Evolution-Discovery Experiment 31).

    `exact_value` should be `magic_entropy(rho)` -- used only to also
    report each snapshot count's mean estimation bias alongside the
    fitted curve, not part of the fit itself.

    Returns `(rows, fit_c, fit_p)`: `rows` is a list of per-snapshot-count
    dicts (`n_snapshots`, `mean_estimate`, `std_estimate`,
    `mean_abs_error`); `fit_c`/`fit_p` are the fitted constants for
    `C / n^p`, usable the same way as `approx_shadow_std` (or pass them to
    `approx_shadow_std`'s formula directly: `fit_c / n ** fit_p`).
    """
    rows = []
    for n_snap in n_snapshots_list:
        estimates = []
        for trial in range(n_trials):
            snaps = sample_classical_shadow(rho, n_snap, seed=seed_base + trial)
            estimates.append(magic_entropy_from_shadows(snaps))
        estimates = np.array(estimates)
        rows.append({
            "n_snapshots": n_snap, "n_trials": n_trials,
            "mean_estimate": float(estimates.mean()), "std_estimate": float(estimates.std()),
            "mean_abs_error": float(abs(estimates.mean() - exact_value)),
        })
    log_n = np.log([r["n_snapshots"] for r in rows])
    log_std = np.log([r["std_estimate"] for r in rows])
    slope, intercept = np.polyfit(log_n, log_std, 1)
    return rows, float(np.exp(intercept)), float(-slope)
