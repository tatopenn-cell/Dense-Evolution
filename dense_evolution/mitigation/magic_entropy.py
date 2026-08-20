"""Magic entropy: a single-qubit density-matrix diagnostic built from the
3-fold self-convolution "Key Unitary" construction (Bu, Gu, Jaffe,
"Stabilizer testing and magic entropy", arXiv:2306.09292, Definitions 7-8).

Originated from a Colab proposal for a pairwise "Quantum Ruzsa Divergence"
(following Bu, Gu, Jaffe, "A convolutional quantum Ruzsa divergence and
its applications", arXiv:2401.14385) that turned out to have no valid
definition for qubits: that paper's pairwise convolution needs
s^2+t^2=1 mod d, which has no solution at d=2. The companion paper above
does not patch this with a qubit-specific pairwise formula -- it defines
a structurally different, minimum-3-input "Key Unitary" convolution (K
quantum registers, K must be ODD, K>=3; there is no K=2 case). For qubits
the smallest valid object is therefore the 3-fold SELF-convolution of one
state with itself, boxtimes_3(rho,rho,rho), and the entropy of its
reduced output register is what the paper calls "magic entropy": zero
for stabilizer states, positive for non-stabilizer ("magic") states (the
paper's Examples 32/33).

Validated end-to-end in Dense-Evolution-Discovery, Experiment 30
(https://tatopenn-cell.github.io/Dense-Evolution-Discovery/quantum_ruzsa_magic_entropy/):
the Key Unitary circuit was checked basis-state by basis-state against
the paper's own Lemma 9 combinatorial identity; magic_entropy was
checked against all six single-qubit stabilizer states (~0, max observed
4e-11) and the two standard magic states T and H (0.811 bits, matching
each other exactly as expected by symmetry); and it was used as a noise
diagnostic that is qualitatively distinct from uhlmann_fidelity and the
sandwiched Renyi divergence (see renyi.py in this same subpackage) --
under amplitude damping it is non-monotonic (rises then returns to
exactly 0, since the p=1 fixed point |0> is a stabilizer state), unlike
either of those two, which change monotonically over the same sweep.

Restricted to SINGLE-QUBIT density matrices (2x2) -- the Key Unitary here
is built for n=1-qubit registers specifically; a multi-qubit
generalization would need a larger Key Unitary circuit (n-qubit
registers, Definition 7 in general) not implemented here.

A shadow-measurement-based estimator for this same quantity, using
randomized measurement snapshots instead of the exact density matrix, is
promoted alongside this module in magic_entropy_shadows.py -- see that
module for why it has its own API shape (measurement snapshots in, not a
density matrix) rather than a function alongside this one.
"""
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["magic_entropy", "magic_entropy_jit"]


def _cnot_matrix(control: int, target: int, n: int = 3) -> np.ndarray:
    """Permutation matrix for a CNOT(control->target) gate on n qubits,
    computational basis ordered as |q0 q1 ... q(n-1)> (this package's
    MSB-first convention, matching dense_evolution.physics.entropy)."""
    dim = 2 ** n
    mat = np.zeros((dim, dim))
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control]:
            bits[target] ^= 1
        j = 0
        for bit in bits:
            j = (j << 1) | bit
        mat[j, i] = 1.0
    return mat


def _build_key_unitary_k3() -> jnp.ndarray:
    """Definition 7 (Bu, Gu, Jaffe, arXiv:2306.09292, p.6), specialized to
    K=3 registers of n=1 qubit each -- the minimum valid case (K must be
    odd). U = (CNOT_{2->1} CNOT_{3->1}) (CNOT_{1->2} CNOT_{1->3}): layer 1
    fans register 1 into registers 2,3; layer 2 XORs registers 2,3 (their
    post-layer-1 values) back into register 1. Verified basis-state by
    basis-state against the paper's own Lemma 9 identity in
    Dense-Evolution-Discovery's tests/test_quantum_ruzsa_magic_entropy.py."""
    layer1 = _cnot_matrix(0, 2) @ _cnot_matrix(0, 1)
    layer2 = _cnot_matrix(2, 0) @ _cnot_matrix(1, 0)
    return jnp.array(layer2 @ layer1, dtype=jnp.complex128)


_KEY_UNITARY_K3 = _build_key_unitary_k3()


def _self_convolve_3_core(rho: jnp.ndarray) -> jnp.ndarray:
    """boxtimes_3(rho,rho,rho) = Tr_{2,3}[V (rho (x) rho (x) rho) V^dagger]
    (Definition 8). `rho` may be pure or mixed. Not built on
    `dense_evolution.physics.entropy.partial_trace` -- that helper is
    pure-statevector-only, and this needs to trace out a general (possibly
    mixed) 3-qubit density matrix built from a possibly-mixed input."""
    rho_full = jnp.kron(jnp.kron(rho, rho), rho)
    evolved = _KEY_UNITARY_K3 @ rho_full @ jnp.conj(_KEY_UNITARY_K3).T
    tensor = evolved.reshape(2, 2, 2, 2, 2, 2)
    return jnp.einsum("ijkljk->il", tensor)


def _magic_entropy_core(rho: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    reduced = _self_convolve_3_core(rho)
    ev = jnp.linalg.eigvalsh(reduced)
    safe_ev = jnp.clip(jnp.real(ev), eps, 1.0)
    return -jnp.sum(safe_ev * jnp.log2(safe_ev))


def magic_entropy(rho: jnp.ndarray) -> float:
    """Magic entropy of a single-qubit density matrix `rho` (2x2), in bits
    (log2) -- NOTE this differs from
    `dense_evolution.physics.entropy.von_neumann_entropy`'s natural-log
    (nats) convention; kept as log2 here to match both the source paper's
    own convention and the values already published in
    Dense-Evolution-Discovery's Experiment 30.

    Zero for every single-qubit stabilizer state (|0>, |1>, |+>, |->,
    |+i>, |-i>), positive for non-stabilizer ("magic") states -- e.g. the
    T-state and H-state both give 0.811 bits.

    Differentiable through `jax.grad`, including at stabilizer states
    where the reduced matrix's eigenvalues are exactly degenerate (e.g.
    the fully mixed state I/2 gives eigenvalues [0.5, 0.5]): unlike
    `uhlmann_fidelity`, which needs a custom `_eigh_degenerate_safe` JVP
    rule because it reconstructs eigenVECTORS (ill-defined in a
    degenerate eigenspace), this function only ever needs eigenVALUES
    (`jnp.linalg.eigvalsh`, no eigenvectors), whose gradient is
    well-defined even at exact degeneracies -- confirmed directly:
    `jax.grad(magic_entropy)` is finite (no NaN) at both the fully mixed
    state and a magic state.
    """
    rho = jnp.asarray(rho, dtype=jnp.complex128)
    return float(_magic_entropy_core(rho))


magic_entropy_jit = jax.jit(_magic_entropy_core)
"""`jax.jit`-compiled entry point for `magic_entropy`. `rho` must already
be `complex128`. Returns a jnp scalar, not a Python `float` -- call
`float(...)` yourself if you need one outside a jitted context."""
