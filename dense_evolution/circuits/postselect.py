"""
Postselection: keep only the branch of the state in which a qubit is found in a
chosen state, then renormalise.

This is the operation in the definition of PostBQP (Aaronson,
quant-ph/0412187, Definition 1): the computation is conditioned on a
measurement outcome that has nonzero probability. It is not unitary, so it is
applied to the statevector directly, as the projector |psi><psi| on the
qubit followed by renormalisation, and the success probability is returned
with the new state.
"""
import numpy as np

from ..config import ensure_x64
from .registry import HAS_JAX

if HAS_JAX:
    import jax.numpy as jnp
    xp = jnp
else:
    xp = np

__all__ = ['postselect', 'POSTSELECT_STATES']

_R = 1 / np.sqrt(2)
POSTSELECT_STATES = {
    '0': np.array([1, 0], dtype=np.complex128),
    '1': np.array([0, 1], dtype=np.complex128),
    '+': np.array([_R, _R], dtype=np.complex128),
    '-': np.array([_R, -_R], dtype=np.complex128),
    '+i': np.array([_R, 1j * _R], dtype=np.complex128),
    '-i': np.array([_R, -1j * _R], dtype=np.complex128),
}


def postselect(sv, n_qubits, qubit, state='1'):
    """
    Project `qubit` onto `state` and renormalise.

    `state` is one of '0', '1', '+', '-', '+i', '-i'. Returns
    `(new_sv, probability)`, where `probability` is the chance of that outcome
    before postselection. Raises ValueError if the probability is zero, since
    postselection is only defined for outcomes that can occur (Aaronson,
    Definition 1, condition (i)).
    """
    if state not in POSTSELECT_STATES:
        raise ValueError(f"state must be one of {list(POSTSELECT_STATES)}, got {state!r}")
    if not 0 <= qubit < n_qubits:
        raise ValueError(f"qubit index out of range for {n_qubits} qubits")
    if HAS_JAX:
        ensure_x64()
    psi = xp.asarray(POSTSELECT_STATES[state])
    t = xp.asarray(sv).reshape(2 ** qubit, 2, 2 ** (n_qubits - qubit - 1))
    overlap = xp.einsum('j,ijk->ik', xp.conj(psi), t)
    probability = float(xp.sum(xp.abs(overlap) ** 2))
    if probability <= 1e-15:
        raise ValueError(f"outcome {state!r} on qubit {qubit} has zero probability")
    projected = xp.einsum('j,ik->ijk', psi, overlap).reshape(-1)
    return projected / np.sqrt(probability), probability
