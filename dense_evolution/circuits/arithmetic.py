"""
Reversible arithmetic on qubit registers, applied to a statevector.

Every operation here is a bijection f on computational basis states, so its
unitary is the permutation U|x> = |f(x)> (Vedral, Barenco, Ekert,
quant-ph/9511018, Sect. II, Eqs. 2-3). It is applied directly as an index
permutation of the statevector, which is exact and costs one gather over the
2^n amplitudes. The gate-level circuits from the papers (Cuccaro et al.
quant-ph/0410184 ripple-carry adder, Draper quant-ph/0008033 transform
adder) are the references the tests compare against.

A register is a list of qubit indices, least significant bit first. Qubit 0
is the most significant bit of the statevector index, as everywhere in the
engine.
"""
import numpy as np

from .registry import HAS_JAX

if HAS_JAX:
    import jax.numpy as jnp
    xp = jnp
else:
    xp = np

__all__ = ['add_registers', 'subtract_registers', 'add_constant']


def _register_values(n_qubits, register):
    idx = np.arange(2 ** n_qubits)
    value = np.zeros_like(idx)
    for k, q in enumerate(register):
        value |= ((idx >> (n_qubits - 1 - q)) & 1) << k
    return idx, value


def _with_register(n_qubits, idx, register, value):
    out = idx.copy()
    for k, q in enumerate(register):
        shift = n_qubits - 1 - q
        out = (out & ~(1 << shift)) | (((value >> k) & 1) << shift)
    return out


def _check(n_qubits, *registers):
    used = [q for r in registers for q in r]
    if len(set(used)) != len(used):
        raise ValueError("registers must not share qubits")
    if any(q < 0 or q >= n_qubits for q in used):
        raise ValueError(f"qubit index out of range for {n_qubits} qubits")
    if any(len(r) == 0 for r in registers):
        raise ValueError("registers must not be empty")


def _permute(sv, target):
    sv = xp.asarray(sv)
    if xp is np:
        out = np.zeros_like(sv)
        out[target] = sv
        return out
    return xp.zeros_like(sv).at[target].set(sv)


def add_registers(sv, n_qubits, a, b):
    """
    |a, b> -> |a, (a + b) mod 2^len(b)>.

    With len(b) = len(a) + 1 the sum never overflows (Vedral et al.,
    Sect. III.A, Eq. 9); with len(b) = len(a) it is addition modulo 2^n
    (Cuccaro et al., Sect. 4.1).
    """
    _check(n_qubits, a, b)
    idx, va = _register_values(n_qubits, a)
    _, vb = _register_values(n_qubits, b)
    target = _with_register(n_qubits, idx, b, (va + vb) % (1 << len(b)))
    return _permute(sv, target)


def subtract_registers(sv, n_qubits, a, b):
    """
    |a, b> -> |a, (b - a) mod 2^len(b)>, the inverse of add_registers.

    If b < a the result is 2^len(b) - (a - b) and, with len(b) = len(a) + 1,
    its most significant qubit is 1 (Vedral et al., Sect. III.A).
    """
    _check(n_qubits, a, b)
    idx, va = _register_values(n_qubits, a)
    _, vb = _register_values(n_qubits, b)
    target = _with_register(n_qubits, idx, b, (vb - va) % (1 << len(b)))
    return _permute(sv, target)


def add_constant(sv, n_qubits, b, c):
    """
    |b> -> |(b + c) mod 2^len(b)> for a classical integer c.

    c = 1 is increment, c = -1 decrement. Same operation as Draper's
    transform adder with classical addend (Sect. 5) and Beauregard's
    phiADD(c) (quant-ph/0205095, Sect. 2.1).
    """
    _check(n_qubits, b)
    idx, vb = _register_values(n_qubits, b)
    target = _with_register(n_qubits, idx, b, (vb + int(c)) % (1 << len(b)))
    return _permute(sv, target)
