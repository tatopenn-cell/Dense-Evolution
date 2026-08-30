"""Stabilizer Renyi Entropy (SRE): a per-STATE nonstabilizerness ("magic")
monotone (Leone, Oliviero, Hamma, "Stabilizer Renyi Entropy",
arXiv:2106.12587, Phys. Rev. Lett. 128, 050402 (2022), Eq. 5-8 there,
labeled Eq. 14/18 in the paper that motivated promoting this).

NOT the same quantity as `dense_evolution.mitigation.magic_entropy`
(Bu-Gu-Jaffe's 3-fold self-convolution "Key Unitary" construction,
single-qubit only) or `sandwiched_renyi_divergence` (Muller-Lennert et
al., a DIVERGENCE between TWO density matrices -- "how different are rho
and sigma", answering a different question entirely; the shared "Renyi"
in both names is a coincidence of both being alpha-generalizations of
entropy applied to different objects, not overlapping math). This SRE is
a genuinely different, MULTI-qubit, SINGLE-state magic monotone: zero for
every stabilizer state, positive otherwise.

M_2(psi) = -log2[ (1/d) * sum_a sum_b |WHT[c_a](b)|^4 ], where
c_a(x) = conj(psi(x)) * psi(x XOR a) and WHT is the length-d Walsh-Hadamard
transform (signmat[b,x] = (-1)^popcount(b AND x)), d = 2**n_qubits.

Verified against known values: every computational-basis/stabilizer state
gives exactly 0; a single T state ((cos(pi/8), sin(pi/8)) in the
computational basis) gives -log2(0.75) = 0.415037 bits, matching the
closed-form derivation of Eq. 5 by hand (not a value copied from
elsewhere).

Promoted from Dense-Evolution-Discovery's wormhole_magic_entropy.py
(2026-08-29), where it was implemented fresh because the existing
magic_entropy is single-qubit only -- this quantity has no dependency on
that use case (wormhole teleportation), so it belongs here as a general
multi-qubit magic diagnostic.

VECTORIZATION NOTE: the Discovery script's original version had an
explicit Python `for a in range(d)` loop, each iteration doing its own
(d,d)@(d,) matrix-vector product -- d sequential small matmuls. This
promoted version instead builds the (d,d) matrix of every c_a(x) pair at
once (broadcasting over x and a together) and applies the Walsh-Hadamard
transform as ONE (d,d)@(d,d) matmul -- same O(d^3) total FLOP count, but
expressed as a single large matmul JAX/XLA can execute efficiently
(GPU/TPU-friendly, no per-iteration Python dispatch overhead), matching
this package's JAX-by-default convention. Still O(d^3), not the paper's
own asymptotically-better O(4^n * n) Walsh-Hadamard butterfly algorithm
(n = log2(d) qubits) -- unimplemented here, same as the Discovery
original; fine for d up to a few thousand (n up to ~11-12 qubits), the
sizes this package's exact-statevector backends already target.
"""
import jax
import jax.numpy as jnp

__all__ = ["stabilizer_renyi_entropy", "stabilizer_renyi_entropy_jit"]


def _stabilizer_renyi_entropy_core(psi):
    d = psi.shape[0]
    idx = jnp.arange(d)
    # Static (trace-time) construction, same style as magic_entropy.py's
    # _cnot_matrix -- one Python loop per distinct d, cached by jax.jit.
    popcount = jnp.array([bin(i).count("1") for i in range(d)])

    bitwise_and = idx[:, None] & idx[None, :]          # [b, x] = b & x
    signmat = (-1.0) ** popcount[bitwise_and]            # Walsh-Hadamard sign matrix

    xor_table = idx[:, None] ^ idx[None, :]              # [x, a] = x ^ a
    c = jnp.conj(psi)[:, None] * psi[xor_table]          # [x, a] = c_a(x)

    wht = signmat @ c                                     # [b, a] = WHT[c_a](b)
    total = jnp.sum(jnp.abs(wht) ** 4)
    return -jnp.log2(total / d)


def stabilizer_renyi_entropy(psi):
    """Stabilizer Renyi Entropy of a pure state `psi` (length 2**n_qubits),
    in bits (log2) -- the paper's own convention.

    Zero for every stabilizer state, positive for non-stabilizer ("magic")
    states -- e.g. a single T state gives 0.415037 bits.

    Parameters
    ----------
    psi : array-like, shape (2**n_qubits,)
        A normalized pure statevector.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> psi0 = np.zeros(8, dtype=complex); psi0[0] = 1.0
    >>> round(stabilizer_renyi_entropy(psi0), 6)  # computational basis state: stabilizer, expect 0
    0.0
    """
    psi = jnp.asarray(psi, dtype=jnp.complex128)
    d = psi.shape[0]
    if d < 1 or (d & (d - 1)) != 0:
        raise ValueError(f"psi length {d} is not a power of 2 (must be 2**n_qubits)")
    return float(_stabilizer_renyi_entropy_core(psi))


stabilizer_renyi_entropy_jit = jax.jit(_stabilizer_renyi_entropy_core)
"""`jax.jit`-compiled entry point for `stabilizer_renyi_entropy`. `psi`
must already be `complex128`. Returns a jnp scalar, not a Python `float`
-- call `float(...)` yourself if you need one outside a jitted context."""
