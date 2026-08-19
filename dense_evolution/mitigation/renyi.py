"""Sandwiched Quantum Renyi Divergence for full density-matrix diagnostics
(Muller-Lennert, Reeb, Wolf, Wilde, "On quantum Renyi entropies: a new
generalization and some applications", arXiv:1306.3142, Definition 1).

D_alpha(rho||sigma) = 1/(alpha-1) * log2 Tr[(sigma^e rho sigma^e)^alpha],
e = (1-alpha)/(2*alpha), with the alpha->1 limit reducing to the standard
quantum relative entropy and alpha=1/2 reducing to a fidelity-based form.

Originated from a Colab proposal with a real bug in its case_general
branch: `tr_inner = jnp.maximum(tr_inner, 1.0)` floors the inner trace at
1.0 even when the true value is < 1 (the normal case for non-commuting
rho, sigma), silently forcing every result to log2(1)=0 -- confirmed
directly in the Colab's own printed output (alpha=1.5 gave exactly
0.000000 across an entire rotation sweep). A second, deeper bug survived
the floor-value fix alone: for alpha > 1, a trace below 1 is not a
numerical artifact to clamp away, it is the genuine signature of a
support mismatch (supp(rho) not contained in supp(sigma)), which the
divergence must report as +inf, not a finite (and wrong-signed) number --
verified by hand on two different pure states: Tr[Q^1.5] = 0.6759,
matching the closed-form prediction (|<sigma|rho>|^2)^alpha exactly, and
plugging that into the naive formula gives a finite NEGATIVE divergence,
worse than the original bug's silent zero since it looks plausible
instead of visibly wrong.

Fixed and validated in Dense-Evolution-Discovery, Experiment 29
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/sandwiched_renyi_density_matrix/):
against the alpha->1 relative-entropy limit (matches an independent numpy
reference to 4 decimal places), the commuting/diagonal case (reduces
exactly to the classical Renyi divergence), and the support-violation
+inf case (verified at alpha>1, confirmed the branch does not fire
spuriously at alpha<1).

Its originally proposed use case -- replacing the JSD-based truncation
criterion in dense_evolution.mps's bond-dimension search -- was
independently disproven: on the diagonal singular-value spectrum used
there, rho and sigma commute, so a non-commuting-aware divergence induces
the exact same truncation ordering as JSD (5 benchmark configurations,
byte-identical chi_used and truncation error every time) -- nothing for
it to add in that setting. Promoted here instead for the genuinely
non-commuting full-density-matrix diagnostic use case it WAS validated
against: alongside uhlmann_fidelity, tracking a Bell state degraded by
amplitude damping, where the two metrics' noise-sensitivity curves
visibly diverge from each other.
"""
import jax
import jax.numpy as jnp

__all__ = ["sandwiched_renyi_divergence", "sandwiched_renyi_divergence_jit"]

_EPS = 1e-12


def _case_half(rho, sigma):
    ev_r, ec_r = jnp.linalg.eigh(rho)
    safe_ev_r = jnp.where(ev_r > _EPS, ev_r, 0.0)
    sqrt_rho = (ec_r * jnp.sqrt(safe_ev_r)) @ jnp.conj(ec_r).T
    uhlmann_mat = sqrt_rho @ sigma @ sqrt_rho
    ev_u = jnp.linalg.eigvalsh(uhlmann_mat)
    safe_ev_u = jnp.where(ev_u > _EPS, ev_u, 0.0)
    fidelity = jnp.clip(jnp.sum(jnp.sqrt(safe_ev_u)), 0.0, 1.0)
    return -2.0 * jnp.log2(jnp.maximum(fidelity, _EPS))


def _case_one(rho, sigma):
    ev_r, ec_r = jnp.linalg.eigh(rho)
    safe_ev_r = jnp.where(ev_r > _EPS, ev_r, 1.0)
    log_rho = (ec_r * jnp.where(ev_r > _EPS, jnp.log2(safe_ev_r), 0.0)) @ jnp.conj(ec_r).T
    ev_s, ec_s = jnp.linalg.eigh(sigma)
    safe_ev_s = jnp.where(ev_s > _EPS, ev_s, 1.0)
    log_sigma = (ec_s * jnp.where(ev_s > _EPS, jnp.log2(safe_ev_s), 0.0)) @ jnp.conj(ec_s).T
    return jnp.trace(rho @ (log_rho - log_sigma)).real


def _case_general(rho, sigma, alpha):
    exponent = (1.0 - alpha) / (2.0 * alpha)
    ev_s, ec_s = jnp.linalg.eigh(sigma)
    mask_s = ev_s > _EPS
    safe_ev_s = jnp.where(mask_s, ev_s, 1.0)
    pow_ev_s = jnp.where(mask_s, safe_ev_s ** exponent, 0.0)
    sigma_pow = (ec_s * pow_ev_s) @ jnp.conj(ec_s).T

    int_m = sigma_pow @ rho @ sigma_pow
    v_int = jnp.linalg.eigvalsh(int_m)
    mask_int = v_int > _EPS
    safe_v_int = jnp.where(mask_int, v_int, 1.0)
    pow_v_int = jnp.where(mask_int, safe_v_int ** alpha, 0.0)

    tr_inner = jnp.sum(pow_v_int)

    # For alpha > 1, D_alpha is finite only when supp(rho) subset supp(sigma);
    # tr_inner < 1 there is the genuine signature of a support mismatch, not
    # a numerical artifact -- see the module docstring's bug-history note.
    is_support_violation = (alpha > 1.0) & (tr_inner < 1.0 - 1e-9)
    tr_inner_safe = jnp.maximum(tr_inner, _EPS)
    finite_result = ((1.0 / (alpha - 1.0)) * jnp.log2(tr_inner_safe)).real
    return jnp.where(is_support_violation, jnp.inf, finite_result)


def _sandwiched_renyi_divergence_core(rho: jnp.ndarray, sigma: jnp.ndarray, alpha: float = 1.5) -> jnp.ndarray:
    # alpha must be a jnp value, not a raw Python float, before it reaches
    # _case_general: jax.lax.cond traces BOTH branches regardless of which
    # one runs, and `1.0 / (alpha - 1.0)` at alpha=1.0 raises a Python
    # ZeroDivisionError if alpha is still a plain float at that point (the
    # case_one branch's own guard never gets a chance to skip it) -- under
    # jax.jit this conversion happens automatically (all arguments get
    # traced), which is why the original un-split Discovery version never
    # hit this; the eager (non-jit) core path here needs it explicitly.
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    is_half = jnp.isclose(alpha, 0.5)
    is_one = jnp.isclose(alpha, 1.0)
    return jax.lax.cond(
        is_half, lambda: _case_half(rho, sigma),
        lambda: jax.lax.cond(is_one, lambda: _case_one(rho, sigma), lambda: _case_general(rho, sigma, alpha)),
    )


def sandwiched_renyi_divergence(rho: jnp.ndarray, sigma: jnp.ndarray, alpha: float = 1.5) -> float:
    """Sandwiched quantum Renyi divergence D_alpha(rho||sigma), in bits
    (log2). `rho`, `sigma` are density matrices of the same dimension;
    `alpha` selects the order (0.5 -> fidelity-based, 1.0 -> standard
    relative entropy, both handled as exact closed-form limits rather
    than through the general formula's own alpha->0.5/1 numerical
    instability).

    Zero when rho == sigma at every alpha (verified). For alpha > 1,
    returns `+inf` when supp(rho) is not contained in supp(sigma) --
    e.g. two different pure (rank-1) states -- rather than a finite
    number; see the module docstring for why this is the mathematically
    correct behavior, not an edge-case failure.

    KNOWN LIMITATION at exactly alpha=1.0: the same support-violation
    check is NOT applied to the alpha=1 (relative-entropy) branch, which
    instead clips log(0)-type contributions to 0 rather than diverging --
    e.g. D_1(rho||sigma) for two different pure states returns 0.0, not
    +inf, even though the true relative entropy diverges there too.
    Experiment 29 validated the alpha=1 branch only against full-rank
    (depolarized) inputs specifically to sidestep this exactly-singular
    case (`scipy.linalg.logm` itself raises `LogmExactlySingularWarning`
    on singular inputs -- an inherent ill-conditioning of relative
    entropy near degenerate support, not unique to this implementation).
    Do not rely on alpha=1 to correctly flag a support mismatch; use
    alpha slightly above 1 (e.g. 1.001) if that matters for your use case.

    Validation-only, like `uhlmann_fidelity`: meant to grade a correction
    against a known reference state, not to feed into one (see
    `uhlmann_fidelity`'s docstring for the full "ideal state as oracle"
    argument, which applies here identically).
    """
    rho = jnp.asarray(rho, dtype=jnp.complex128)
    sigma = jnp.asarray(sigma, dtype=jnp.complex128)
    return float(_sandwiched_renyi_divergence_core(rho, sigma, alpha))


sandwiched_renyi_divergence_jit = jax.jit(_sandwiched_renyi_divergence_core)
"""`jax.jit`-compiled entry point for `sandwiched_renyi_divergence`. `rho`/
`sigma` must already be `complex128`. Returns a jnp scalar, not a Python
`float`."""
