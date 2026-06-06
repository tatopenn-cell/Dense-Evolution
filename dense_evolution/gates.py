import numpy as np
from .registry import HAS_JAX

<<<<<<< HEAD
# Scegli il backend corretto una volta sola
=======

>>>>>>> 10dd0b7 (v8.1.2 - SafeMemoryGuard Anti-OOM, chunk.py rewrite, README update)
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
    'rx': 9, 'ry': 10, 'rz': 11, 'cx': 20, 'cz': 21
}

<<<<<<< HEAD
# Un dizionario unico, pulito e senza codice duplicato
=======

>>>>>>> 10dd0b7 (v8.1.2 - SafeMemoryGuard Anti-OOM, chunk.py rewrite, README update)
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
