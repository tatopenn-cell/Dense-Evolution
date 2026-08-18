import numpy as np
from .registry import HAS_JAX


if HAS_JAX:
    import jax.numpy as jnp
    xp = jnp
else:
    xp = np

INV2 = 1.0 / np.sqrt(2.0)

# Ora i gate statici usano il backend nativo (Numpy o JAX/GPU)
GATES = {
    'h': INV2 * xp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex),
    'x': xp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'y': xp.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    'z': xp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    's': xp.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),
    'sdg': xp.array([[1.0, 0.0], [0.0, -1j]], dtype=complex),
    't': xp.array([[1.0, 0.0], [0.0, xp.exp(1j * np.pi / 4)]], dtype=complex),
    'tdg': xp.array([[1.0, 0.0], [0.0, xp.exp(-1j * np.pi / 4)]], dtype=complex),
    'sx': 0.5 * xp.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    'id': xp.eye(2, dtype=complex),
    'cx': xp.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]], dtype=complex),
    'cz': xp.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0,-1]], dtype=complex),
    'cy': xp.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0,-1j],[0, 0, 1j, 0]], dtype=complex),
    'swap': xp.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]], dtype=complex),
    'iswap': xp.array([[1, 0, 0, 0],[0, 0, 1j, 0],[0, 1j, 0, 0],[0, 0, 0, 1]], dtype=complex),
    'ecr': INV2 * xp.array([[0, 0, 1, 1j],[0, 0, 1j, 1],[1,-1j, 0, 0],[-1j, 1, 0, 0]], dtype=complex),
    'ccx': xp.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]], dtype=complex)
}

GATE_IDS = {
    'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4, 's': 5, 'sdg': 6, 't': 7, 'tdg': 8,
    'rx': 9, 'ry': 10, 'rz': 11,
    'p': 12, 'u1': 12, 'phase': 12,   # already had a kernel entry (index 12), just no name reached it
    'sx': 13,
    'cx': 20, 'cz': 21, 'cp': 22, 'cphase': 22,
    # 23 = swap, reserved (never dispatched here: QuantumTranspiler.transpile
    # always decomposes 'swap' into 3xCX before a gate name reaches this table)
    'cy': 24, 'crz': 25,
}


# Which PARAMETRIC_GATES entries act on 2 qubits (vs. every other entry,
# which acts on 1). Needed to dispatch correctly: a 1-qubit gate with 2
# params (u2: phi, lam) and a 2-qubit gate with 1 param (cp/crz: theta)
# both produce a 3-element args tuple downstream, so arg *count* alone
# can't disambiguate them -- BUG FIX: simulator.py's run_circuit used to
# dispatch PARAMETRIC_GATES purely on len(args), which silently mis-typed
# 'u2' as a 2-qubit gate (crashing with a TypeError from calling its
# 2-param lambda with only 1 positional argument) since it collided with
# cp/crz's own 3-element shape. Named lookup here, not arg-count
# guessing, is the fix.
_TWO_QUBIT_PARAMETRIC_GATES = frozenset(('cp', 'cphase', 'crz'))

PARAMETRIC_GATES = {
    'rx': lambda theta: xp.array([[xp.cos(theta/2), -1j*xp.sin(theta/2)], [-1j*xp.sin(theta/2), xp.cos(theta/2)]], dtype=complex),
    'ry': lambda theta: xp.array([[xp.cos(theta/2), -xp.sin(theta/2)], [xp.sin(theta/2), xp.cos(theta/2)]], dtype=complex),
    'rz': lambda theta: xp.array([[xp.exp(-1j*theta/2), 0.0], [0.0, xp.exp(1j*theta/2)]], dtype=complex),
    'cp': lambda lam: xp.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, xp.exp(1j*lam)]], dtype=complex),
    'crz': lambda theta: xp.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, xp.exp(-1j*theta/2), 0],[0, 0, 0, xp.exp(1j*theta/2)]], dtype=complex),
    'u3': lambda theta, phi, lam: xp.array([[xp.cos(theta/2), -xp.exp(1j*lam)*xp.sin(theta/2)], [xp.exp(1j*phi)*xp.sin(theta/2), xp.exp(1j*(phi+lam))*xp.cos(theta/2)]], dtype=complex),
    'u2': lambda phi, lam: xp.array([[1.0, -xp.exp(1j*lam)], [xp.exp(1j*phi), xp.exp(1j*(phi+lam))]], dtype=complex) * INV2,
    'u1': lambda lam: xp.array([[1.0, 0.0], [0.0, xp.exp(1j*lam)]], dtype=complex),
    'p': lambda lam: xp.array([[1.0, 0.0], [0.0, xp.exp(1j*lam)]], dtype=complex)
}
