import numpy as np
from registry import HAS_JAX

if HAS_JAX:
    import jax.numpy as jnp

INV2 = 1.0 / np.sqrt(2.0)

GATES = {
    'h': INV2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex),
    'x': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'y': np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    'z': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    's': np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),
    'sdg': np.array([[1.0, 0.0], [0.0, -1j]], dtype=complex),
    't': np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=complex),
    'tdg': np.array([[1.0, 0.0], [0.0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    'sx': 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    'id': np.eye(2, dtype=complex),
    'cx': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex),
    'cz': np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex),
    'cy': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]], dtype=complex),
    'swap': np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex),
    'iswap': np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex),
    'ecr': INV2 * np.array([[0,0,1,1j],[0,0,1j,1],[1,-1j,0,0],[-1j,1,0,0]], dtype=complex),
    'ccx': np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]], dtype=complex)
}

GATE_IDS = {
    'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4, 's': 5, 'sdg': 6, 't': 7, 'tdg': 8,
    'rx': 9, 'ry': 10, 'rz': 11, 'cx': 20, 'cz': 21
}

def _build_parametric_gates():
    if HAS_JAX:
        return {
            'rx': lambda theta: jnp.array([[jnp.cos(theta/2), -1j*jnp.sin(theta/2)], [-1j*jnp.sin(theta/2), jnp.cos(theta/2)]], dtype=complex),
            'ry': lambda theta: jnp.array([[jnp.cos(theta/2), -jnp.sin(theta/2)], [jnp.sin(theta/2), jnp.cos(theta/2)]], dtype=complex),
            'rz': lambda theta: jnp.array([[jnp.exp(-1j*theta/2), 0.0], [0.0, jnp.exp(1j*theta/2)]], dtype=complex),
            'cp': lambda lam: jnp.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,jnp.exp(1j*lam)]], dtype=complex),
            'crz': lambda theta: jnp.array([[1,0,0,0],[0,1,0,0],[0,0,jnp.exp(-1j*theta/2),0],[0,0,0,jnp.exp(1j*theta/2)]], dtype=complex),
            'u3': lambda theta, phi, lam: jnp.array([[jnp.cos(theta/2), -jnp.exp(1j*lam)*jnp.sin(theta/2)], [jnp.exp(1j*phi)*jnp.sin(theta/2), jnp.exp(1j*(phi+lam))*jnp.cos(theta/2)]], dtype=complex),
            'u2': lambda phi, lam: jnp.array([[1.0, -jnp.exp(1j*lam)], [jnp.exp(1j*phi), jnp.exp(1j*(phi+lam))]], dtype=complex) * INV2,
            'u1': lambda lam: jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j*lam)]], dtype=complex),
            'p': lambda lam: jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j*lam)]], dtype=complex)
        }
    return {
        'rx': lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex),
        'ry': lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=complex),
        'rz': lambda theta: np.array([[np.exp(-1j*theta/2), 0.0], [0.0, np.exp(1j*theta/2)]], dtype=complex),
        'cp': lambda lam: np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(1j*lam)]], dtype=complex),
        'crz': lambda theta: np.array([[1,0,0,0],[0,1,0,0],[0,0,np.exp(-1j*theta/2),0],[0,0,0,np.exp(1j*theta/2)]], dtype=complex),
        'u3': lambda theta, phi, lam: np.array([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]], dtype=complex),
        'u2': lambda phi, lam: np.array([[1.0, -np.exp(1j*lam)], [np.exp(1j*phi), np.exp(1j*(phi+lam))]], dtype=complex) * INV2,
        'u1': lambda lam: np.array([[1.0, 0.0], [0.0, np.exp(1j*lam)]], dtype=complex),
        'p': lambda lam: np.array([[1.0, 0.0], [0.0, np.exp(1j*lam)]], dtype=complex)
    }

PARAMETRIC_GATES = _build_parametric_gates()


