"""Classical Kullback-Leibler divergence between probability distributions
(Kullback, S. & Leibler, R.A., "On Information and Sufficiency", The
Annals of Mathematical Statistics, 22(1), 79-86, 1951).

D_KL(p||q) = sum_x p(x) * log2(p(x)/q(x)), the relative entropy of q from
p, in bits (log2, matching the rest of this subpackage's convention --
`sandwiched_renyi_divergence`/`magic_entropy` both use log2).

Checked against the paper's own text directly (Section 2, eq. 2.2-2.3),
not assumed from the textbook formula alone: what this module implements
is what Kullback & Leibler call I(1:2), "the mean information for
discrimination between H1 and H2" -- what the broader literature later
popularized as "the KL divergence". Their OWN word "divergence",
J(1,2) = I(1:2) + I(2:1) (eq. 2.9), names the symmetrized sum of both
directions instead -- deliberately not implemented here, since it would
duplicate the Jensen-Shannon divergence this codebase already uses
(`mps.py`'s bond-dimension search, `zne.py`'s predictive ZNE), which is
bounded and better-behaved at disjoint supports.

Distinct in kind from `sandwiched_renyi_divergence(rho, sigma, alpha=1.0)`,
which reduces to the QUANTUM relative entropy Tr[rho(log rho - log sigma)]
between density MATRICES via matrix logarithms -- this module implements
the simpler classical case directly over probability VECTORS (e.g. a
measurement-outcome distribution |psi|^2, or any other normalized
histogram), with no eigendecomposition needed.

Originated from an honest gap flagged in this subpackage's own healing.py
docstring: calculate_vettore_dinamico's core term, log(E_B/E_A), is a
single un-weighted log-likelihood ratio between two scalars -- the same
elementary quantity this divergence is built from, but not this
divergence itself. Built and validated in Dense-Evolution-Discovery,
Experiment 32
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kullback_leibler_divergence/):
against an independent scipy.stats.entropy reference (1e-9 bits across 20
random trials), Gibbs' inequality (D_KL >= 0, 200 random pairs, never
negative), a genuine support-violation case (+inf, not a finite wrong
number), and a real measurement-distribution application confirming this
is not a trivial rescaling of healing.py's existing scalar signal.

Additive, not a replacement for the already-validated healing pipeline.
"""
import jax
import jax.numpy as jnp

__all__ = ["kl_divergence", "kl_divergence_jit"]

_EPS = 1e-12


def _kl_divergence_core(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    # 0 * log(0/q) = 0 by the standard x*log(x) -> 0 (x -> 0+) convention
    # (Cover & Thomas, "Elements of Information Theory", 2nd ed., section
    # 2.1) -- terms where p is (numerically) zero contribute nothing,
    # regardless of q there.
    p_is_zero = p < _EPS

    # A term is only a genuine support violation when p(x) > 0 but
    # q(x) == 0: p has mass where q has none, so no finite log-ratio can
    # account for it -- D_KL(p||q) is +inf by definition, not a numerical
    # artifact to clamp away (the same lesson already learned the hard way
    # for sandwiched_renyi_divergence's alpha>1 case -- see renyi.py).
    q_is_zero = q < _EPS
    is_support_violation = jnp.any((~p_is_zero) & q_is_zero)

    safe_p = jnp.where(p_is_zero, 1.0, p)
    safe_q = jnp.where(q_is_zero, 1.0, q)
    terms = jnp.where(p_is_zero, 0.0, safe_p * jnp.log2(safe_p / safe_q))
    finite_result = jnp.sum(terms)
    return jnp.where(is_support_violation, jnp.inf, finite_result)


def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> float:
    """Classical Kullback-Leibler divergence D_KL(p||q), in bits.

    `p`, `q` are 1-D real, non-negative probability vectors of the same
    length, each summing to 1 (not validated here -- callers pass in a
    normalized distribution, e.g. `jnp.abs(psi) ** 2` for a statevector's
    measurement-outcome probabilities). Not symmetric: D_KL(p||q) !=
    D_KL(q||p) in general.

    Zero iff p == q (Gibbs' inequality: D_KL(p||q) >= 0 always, with
    equality only at p == q). Returns `+inf` when p has support where q
    does not (p(x) > 0, q(x) == 0 for some x) -- the correct value, not an
    edge case to avoid; see the module docstring.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    return float(_kl_divergence_core(p, q))


kl_divergence_jit = jax.jit(_kl_divergence_core)
"""`jax.jit`-compiled entry point for `kl_divergence`. `p`/`q` must already
be `float64` arrays. Returns a jnp scalar, not a Python `float`."""
