"""Coherent, differentiable adversarial noise for stabilizer codes.

Promoted from Dense-Evolution-Discovery's Steane [[7,1,3]] investigation
(scripts/steane_code_block2_adversarial_noise.py and
steane_code_block3_linf_constrained_attack.py), generalized here from a
hard-coded 7-qubit Steane table to any stabilizer list, matching this
module's own `dense_evolution.qec.compute_syndrome` conventions (equal-
length Pauli strings over IXYZ, qubit q = character q).

Unlike `dense_evolution.registry.NoiseModel` (stochastic Kraus channels,
sampled via a hard threshold that is not usefully differentiable), this
module attacks a genuinely continuous channel: independent per-qubit
coherent over-rotations rz(delta_q) applied to all qubits at once --
the natural differentiable analogue of physical miscalibration. Because
it is JAX-differentiable end-to-end, `craft_adversarial_delta` can
gradient-search for a worst-case coherent-error direction instead of
only sampling random ones.

Honest negative result, kept because it is the actual finding, not
smoothed over: unconstrained (`craft_adversarial_delta`, L2-ball only)
PGD on `x_stabilizer_leakage` degenerately concentrates the whole error
budget on whichever single qubit is shared by every X-stabilizer
generator. A coherent rz error on one qubit alone always collapses to
weight-0 or weight-1, which a distance-3 code's decoder corrects
exactly every time -- so the "adversarial" direction found this way has
ZERO real decoder-failure rate, while random multi-qubit directions of
the same L2 budget fail readily. `craft_adversarial_delta_constrained`
fixes this by also capping the per-qubit angle (L-infinity), forcing
the search to spread the budget across multiple qubits -- the regime
that actually causes failures. Use the constrained version if the goal
is a genuine worst-case test; the unconstrained version is kept because
it is a real, useful example of a differentiable proxy objective being
anti-correlated with the thing it was meant to approximate.
"""
import numpy as np
import jax
import jax.numpy as jnp

from dense_evolution.physics.qec import compute_syndrome

__all__ = [
    "apply_rz_all",
    "x_stabilizer_leakage",
    "craft_adversarial_delta",
    "project_l2_linf",
    "craft_adversarial_delta_constrained",
    "decoder_failure_rate",
    "random_delta_failure_stats",
]


def _flip_mask(pauli_str: str) -> int:
    """Bitmask of qubits carrying an X or Y term, qubit q = bit (n-1-q) --
    dense_evolution's own MSB-first statevector-index convention."""
    n = len(pauli_str)
    mask = 0
    for q, p in enumerate(pauli_str):
        if p in ('X', 'Y'):
            mask |= 1 << (n - 1 - q)
    return mask


def apply_rz_all(sv0: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Coherent per-qubit rz(delta_q) applied to every qubit of `sv0` at
    once. rz gates are diagonal and all commute, so this is exact
    elementwise phase multiplication, not a per-gate circuit simulation
    -- and fully JAX-differentiable in `delta`.

    Parameters
    ----------
    sv0 : statevector, length 2**n_qubits
    delta : real array, length n_qubits -- rz angle for each qubit

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> sv0 = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    >>> sv1 = apply_rz_all(sv0, jnp.array([np.pi]))
    >>> round(float(jnp.abs(sv1[0]) ** 2), 6)
    1.0
    """
    n = delta.shape[0]
    dim = sv0.shape[0]
    idx = jnp.arange(dim, dtype=jnp.int32)
    bit_pos = jnp.arange(n - 1, -1, -1, dtype=jnp.int32)
    bits = (idx[:, None] >> bit_pos[None, :]) & 1
    s = (1 - 2 * bits).astype(jnp.float64)
    phase_arg = -0.5 * jnp.sum(delta[None, :] * s, axis=1)
    return sv0 * jnp.exp(1j * phase_arg)


def x_stabilizer_leakage(delta: jnp.ndarray, sv0: jnp.ndarray, stabilizers) -> jnp.ndarray:
    """Total leakage of `sv0`, after a coherent `apply_rz_all(sv0, delta)`
    perturbation, out of the +1 joint eigenspace of `stabilizers` --
    sum_i (1 - <stabilizer_i>) / 2, one term per generator. A smooth,
    bounded ([0, len(stabilizers)]) proxy for how much the coherent
    error disturbs the syndrome; NOT the actual decoder failure
    probability (see module docstring for the documented gap between
    the two). `stabilizers` should be X-type generators to pair with an
    rz (Z-type-diagonal) coherent error -- the same reasoning
    `dense_evolution.qec.compute_syndrome` uses generically for any
    Pauli-string stabilizer list.
    """
    sv = apply_rz_all(sv0, delta)
    dim = sv0.shape[0]
    idx = jnp.arange(dim, dtype=jnp.int32)
    total = 0.0
    for g in stabilizers:
        mask = _flip_mask(g)
        src = idx ^ mask
        expectation = jnp.real(jnp.sum(jnp.conj(sv) * sv[src]))
        total = total + (1.0 - expectation) / 2.0
    return total


def craft_adversarial_delta(sv0: jnp.ndarray, stabilizers, epsilon: float,
                             n_steps: int = 150, step_size: float = 0.05, seed: int = 0):
    """Gradient-ascent PGD on `x_stabilizer_leakage`, projected into the L2
    epsilon-ball around delta=0 after every step. Returns (best_delta as
    numpy, best_leakage, leakage_history).

    See the module docstring: on a distance-3 code this unconstrained
    search finds a direction with real decoder-failure rate 0 -- use
    `craft_adversarial_delta_constrained` for a genuine worst-case test.
    """
    n_qubits = len(stabilizers[0])
    leakage_grad = jax.grad(lambda d, sv: x_stabilizer_leakage(d, sv, stabilizers), argnums=0)

    rng = np.random.default_rng(seed)
    init_dir = rng.normal(size=n_qubits)
    init_dir /= np.linalg.norm(init_dir)
    init_norm = min(epsilon, 1e-2)
    delta = jnp.array(init_dir * init_norm)

    best_delta = delta
    best_leakage = float(x_stabilizer_leakage(delta, sv0, stabilizers))
    history = [best_leakage]

    for _ in range(n_steps):
        grad = leakage_grad(delta, sv0)
        grad_norm = jnp.linalg.norm(grad)
        step = jnp.where(grad_norm > 1e-12, grad / grad_norm, jnp.zeros_like(grad))
        delta = delta + step_size * step
        delta_norm = jnp.linalg.norm(delta)
        delta = jnp.where(delta_norm > epsilon, delta / delta_norm * epsilon, delta)

        current = float(x_stabilizer_leakage(delta, sv0, stabilizers))
        history.append(current)
        if current > best_leakage:
            best_leakage = current
            best_delta = delta

    return np.asarray(best_delta), best_leakage, history


def project_l2_linf(y: np.ndarray, epsilon: float, linf_cap: float, n_bisect: int = 60) -> np.ndarray:
    """Exact projection of `y` onto the intersection of an L2 ball of
    radius `epsilon` and an L-infinity ball (box) of radius `linf_cap` --
    not the same as clip-then-rescale, which can push coordinates back
    outside the box. Box-clips first; if that's already inside the L2
    ball it's the exact answer, otherwise bisects the Lagrange
    multiplier on the L2 constraint until the clipped, rescaled point
    lands exactly on the L2 boundary.

    Examples
    --------
    >>> import numpy as np
    >>> project_l2_linf(np.array([10.0, 0.0]), epsilon=1.0, linf_cap=5.0).round(4)
    array([1., 0.])
    >>> project_l2_linf(np.array([0.1, 0.1]), epsilon=1.0, linf_cap=0.05).round(4)
    array([0.05, 0.05])
    """
    y = np.asarray(y, dtype=np.float64)
    box = np.clip(y, -linf_cap, linf_cap)
    if np.linalg.norm(box) <= epsilon + 1e-12:
        return box
    lo, hi = 0.0, 1e8
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        z = np.clip(y / (1.0 + mid), -linf_cap, linf_cap)
        if np.linalg.norm(z) > epsilon:
            lo = mid
        else:
            hi = mid
    return np.clip(y / (1.0 + hi), -linf_cap, linf_cap)


def craft_adversarial_delta_constrained(sv0: jnp.ndarray, stabilizers, epsilon: float, linf_cap: float,
                                         n_steps: int = 150, step_size: float = 0.05, seed: int = 0):
    """Same PGD search as `craft_adversarial_delta`, but each step is
    projected into the L2-epsilon-ball INTERSECTED with an L-infinity
    box of radius `linf_cap` (via `project_l2_linf`) instead of the L2
    ball alone. Capping the per-qubit angle forbids the degenerate
    one-qubit-takes-everything solution and forces the search to spread
    the budget across multiple qubits -- the regime that actually causes
    real decoder failures (see module docstring)."""
    n_qubits = len(stabilizers[0])
    leakage = lambda d, sv: x_stabilizer_leakage(d, sv, stabilizers)
    leakage_grad = jax.grad(leakage, argnums=0)

    rng = np.random.default_rng(seed)
    init_dir = rng.normal(size=n_qubits)
    init_dir /= np.linalg.norm(init_dir)
    init_norm = min(epsilon, 1e-2)
    delta_np = project_l2_linf(init_dir * init_norm, epsilon, linf_cap)
    delta = jnp.array(delta_np)

    best_delta = delta
    best_leakage = float(leakage(delta, sv0))
    history = [best_leakage]

    for _ in range(n_steps):
        grad = leakage_grad(delta, sv0)
        grad_norm = jnp.linalg.norm(grad)
        step = jnp.where(grad_norm > 1e-12, grad / grad_norm, jnp.zeros_like(grad))
        delta_raw = np.asarray(delta) + step_size * np.asarray(step)
        delta_np = project_l2_linf(delta_raw, epsilon, linf_cap)
        delta = jnp.array(delta_np)

        current = float(leakage(delta, sv0))
        history.append(current)
        if current > best_leakage:
            best_leakage = current
            best_delta = delta

    return np.asarray(best_delta), best_leakage, history


def decoder_failure_rate(delta_np: np.ndarray, sv0_np: np.ndarray, decode_fn,
                          n_trials: int, rng: np.random.Generator) -> float:
    """Real, discrete evaluation of how often a coherent error `delta_np`
    actually fools a decoder -- as opposed to `x_stabilizer_leakage`'s
    smooth proxy. `decode_fn(sv_noisy, rng) -> sv_corrected` is any
    projective-measurement-based decoder (e.g. built around
    `dense_evolution.qec.compute_syndrome`); this function applies the
    coherent error once, then calls `decode_fn` `n_trials` times (the
    syndrome measurement that collapses the coherently-perturbed state
    is itself stochastic, so repeated trials are meaningful even though
    `delta_np` is fixed), and reports the fraction that fail to recover
    `sv0_np` exactly."""
    sv_delta = np.asarray(apply_rz_all(jnp.array(sv0_np), jnp.array(delta_np)))
    n_fail = 0
    for _ in range(n_trials):
        sv_corrected = decode_fn(sv_delta.copy(), rng)
        fidelity = np.abs(np.vdot(sv_corrected, sv0_np)) ** 2
        if fidelity < 1.0 - 1e-6:
            n_fail += 1
    return n_fail / n_trials


def random_delta_failure_stats(sv0_np: np.ndarray, decode_fn, epsilon: float, n_qubits: int,
                                n_random: int, n_trials_each: int, rng: np.random.Generator) -> np.ndarray:
    """Same evaluation as `decoder_failure_rate`, but over `n_random`
    random directions of L2 norm `epsilon` instead of one crafted
    `delta` -- the baseline `craft_adversarial_delta`'s result should be
    compared against (see module docstring: the unconstrained crafted
    direction can score WORSE than this random baseline)."""
    rates = np.zeros(n_random)
    for i in range(n_random):
        d = rng.normal(size=n_qubits)
        d = d / np.linalg.norm(d) * epsilon
        rates[i] = decoder_failure_rate(d, sv0_np, decode_fn, n_trials_each, rng)
    return rates
