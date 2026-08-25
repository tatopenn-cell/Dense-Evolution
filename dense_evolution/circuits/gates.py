import numpy as np
from .registry import HAS_JAX


if HAS_JAX:
    import jax.numpy as jnp
    xp = jnp
else:
    xp = np

INV2 = 1.0 / np.sqrt(2.0)

# BUG FIX: these used to be built with `xp.array(..., dtype=complex)`
# (xp = jax.numpy) -- at MODULE IMPORT time. If jax_enable_x64 isn't
# already True at that exact moment (a real, common case: nothing forces
# it before this import anymore, see dense_evolution/config.py and the
# registry.py HARDWARE_REGISTRY removal), JAX silently truncates
# `dtype=complex` (complex128) to complex64 right here, permanently --
# these are plain module-level constants, computed once, never rebuilt
# later even after ensure_x64() turns x64 on for real simulation. That
# produced a real, measurable bug: gate application mixed a complex128
# statevector with complex64-truncated gate matrices, breaking unitarity
# by ~1e-8 (verified directly: tests/unit/test_simulator.py's norm-
# preservation checks, which require < 1e-12, failed at exactly this
# scale). Built with plain NumPy instead -- never subject to JAX's
# process-wide x64 flag, so these are always genuinely complex128
# regardless of import order; JAX correctly canonicalizes them to
# whatever precision is actually active at the point they're first fed
# into a JAX operation (by then, ensure_x64() has already run). Unlike
# GATES, PARAMETRIC_GATES below is unaffected: its entries are lambdas,
# evaluated lazily at gate-application time (after ensure_x64()), not at
# import time.
GATES = {
    'h': INV2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128),
    'x': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    'y': np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    'z': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    's': np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128),
    'sdg': np.array([[1.0, 0.0], [0.0, -1j]], dtype=np.complex128),
    't': np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    'tdg': np.array([[1.0, 0.0], [0.0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128),
    'sx': 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128),
    'id': np.eye(2, dtype=np.complex128),
    'cx': np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]], dtype=np.complex128),
    'cz': np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0,-1]], dtype=np.complex128),
    'cy': np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0,-1j],[0, 0, 1j, 0]], dtype=np.complex128),
    'swap': np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]], dtype=np.complex128),
    'iswap': np.array([[1, 0, 0, 0],[0, 0, 1j, 0],[0, 1j, 0, 0],[0, 0, 0, 1]], dtype=np.complex128),
    'ecr': INV2 * np.array([[0, 0, 1, 1j],[0, 0, 1j, 1],[1,-1j, 0, 0],[-1j, 1, 0, 0]], dtype=np.complex128),
    'ccx': np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]], dtype=np.complex128)
}

GATE_IDS = {
    'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4, 's': 5, 'sdg': 6, 't': 7, 'tdg': 8,
    'rx': 9, 'ry': 10, 'rz': 11,
    'p': 12, 'u1': 12, 'phase': 12,   # already had a kernel entry (index 12), just no name reached it
    'sx': 13,
    'gphase': 14,   # e^{ia}*I, scalar phase on the whole state -- only emitted by QuantumTranspiler.decompose_u3 (U2/U3 -> Rz/Ry/Rz/GPhase)
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
    'p': lambda lam: xp.array([[1.0, 0.0], [0.0, xp.exp(1j*lam)]], dtype=complex),
    # e^{ia}*I -- a scalar phase on the whole state (see GATE_IDS['gphase']
    # and QuantumTranspiler.decompose_u3 for why this exists).
    'gphase': lambda alpha: xp.exp(1j * alpha) * xp.eye(2, dtype=complex),
}
