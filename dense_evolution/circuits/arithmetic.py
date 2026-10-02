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

__all__ = ['add_registers', 'subtract_registers', 'add_constant',
           'compare_registers', 'compare_constant',
           'add_registers_mod', 'add_constant_mod', 'multiply_add_mod',
           'multiply_mod', 'power_mod']

_COMPARISONS = {
    '<': np.less, '>': np.greater, '<=': np.less_equal,
    '>=': np.greater_equal, '==': np.equal, '!=': np.not_equal,
}


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


def _flip_if(n_qubits, idx, out, condition):
    if not 0 <= out < n_qubits:
        raise ValueError(f"qubit index out of range for {n_qubits} qubits")
    return idx ^ (condition.astype(idx.dtype) << (n_qubits - 1 - out))


def _comparison(op):
    if op not in _COMPARISONS:
        raise ValueError(f"op must be one of {sorted(_COMPARISONS)}, got {op!r}")
    return _COMPARISONS[op]


def compare_registers(sv, n_qubits, a, b, out, op='<'):
    """
    |a, b, z> -> |a, b, z XOR [a op b]>, inputs unchanged.

    op = '<' is the comparator of Cuccaro et al. (Sect. 4.3): the high bit of
    a - b, which is 1 if and only if a < b. The other comparisons follow by
    swapping the operands and negating the output qubit.
    """
    _check(n_qubits, a, b, [out])
    idx, va = _register_values(n_qubits, a)
    _, vb = _register_values(n_qubits, b)
    return _permute(sv, _flip_if(n_qubits, idx, out, _comparison(op)(va, vb)))


def compare_constant(sv, n_qubits, b, c, out, op='<'):
    """
    |b, z> -> |b, z XOR [b op c]> for a classical integer c.

    op = '<' is the most significant qubit after the reverse phiADD(c) of
    Beauregard (quant-ph/0205095, Sect. 2.1, Fig. 4).
    """
    _check(n_qubits, b, [out])
    idx, vb = _register_values(n_qubits, b)
    return _permute(sv, _flip_if(n_qubits, idx, out, _comparison(op)(vb, int(c))))


def _active(n_qubits, idx, controls):
    on = np.ones(idx.shape, dtype=bool)
    for q in controls:
        on &= ((idx >> (n_qubits - 1 - q)) & 1).astype(bool)
    return on


def _check_modulus(N, *registers):
    if N < 1 or any(N > (1 << len(r)) for r in registers):
        raise ValueError(f"modulus N={N} must be >= 1 and fit in the register")


def add_registers_mod(sv, n_qubits, a, b, N):
    """
    |a, b> -> |a, (a + b) mod N> for 0 <= a, b < N (Vedral et al., Sect. III.B,
    Eq. 10). Basis states with a >= N or b >= N are left unchanged, which
    keeps the map a permutation.
    """
    _check(n_qubits, a, b)
    _check_modulus(N, a, b)
    idx, va = _register_values(n_qubits, a)
    _, vb = _register_values(n_qubits, b)
    ok = (va < N) & (vb < N)
    return _permute(sv, _with_register(n_qubits, idx, b, np.where(ok, (va + vb) % N, vb)))


def add_constant_mod(sv, n_qubits, b, c, N, controls=()):
    """
    |b> -> |(b + c) mod N> for 0 <= b < N, applied only where every control
    qubit is 1: the doubly controlled phiADD(c)MOD(N) of Beauregard
    (quant-ph/0205095, Sect. 2.2, Fig. 5). Basis states with b >= N are left
    unchanged.
    """
    _check(n_qubits, b, *([list(controls)] if controls else []))
    _check_modulus(N, b)
    idx, vb = _register_values(n_qubits, b)
    ok = _active(n_qubits, idx, controls) & (vb < N)
    return _permute(sv, _with_register(n_qubits, idx, b, np.where(ok, (vb + int(c)) % N, vb)))


def multiply_add_mod(sv, n_qubits, x, b, a, N, controls=()):
    """
    |x, b> -> |x, (b + a*x) mod N> for 0 <= b < N, applied only where every
    control qubit is 1: CMULT(a)MOD(N) of Beauregard (Sect. 2.3, Fig. 6),
    built in the paper from n controlled modular additions of 2^i*a mod N
    (Vedral et al., Sect. III.C). Basis states with b >= N are left unchanged.
    """
    _check(n_qubits, x, b, *([list(controls)] if controls else []))
    _check_modulus(N, b)
    idx, vx = _register_values(n_qubits, x)
    _, vb = _register_values(n_qubits, b)
    ok = _active(n_qubits, idx, controls) & (vb < N)
    new = (vb + (int(a) % N) * (vx % N)) % N
    return _permute(sv, _with_register(n_qubits, idx, b, np.where(ok, new, vb)))


def multiply_mod(sv, n_qubits, x, a, N, controls=()):
    """
    |x> -> |a*x mod N> in place for 0 <= x < N and gcd(a, N) = 1, applied only
    where every control qubit is 1 (Vedral et al., Sect. II, Eqs. 4-6;
    Beauregard, Sect. 2.3, Fig. 7). With N = 2^len(x) and odd a this is plain
    multiplication modulo 2^n. Basis states with x >= N are left unchanged.
    """
    _check(n_qubits, x, *([list(controls)] if controls else []))
    _check_modulus(N, x)
    if np.gcd(int(a), N) != 1:
        raise ValueError(f"a={a} and N={N} must be coprime for an in-place multiplication")
    idx, vx = _register_values(n_qubits, x)
    ok = _active(n_qubits, idx, controls) & (vx < N)
    return _permute(sv, _with_register(n_qubits, idx, x, np.where(ok, (int(a) * vx) % N, vx)))


def power_mod(sv, n_qubits, x, y, a, N):
    """
    |x, y> -> |x, y * a^x mod N> for 0 <= y < N and gcd(a, N) = 1. With y = 1
    this is the modular exponentiation |x, 1> -> |x, a^x mod N> of Shor's
    algorithm (Vedral et al., Eq. 1 and Sect. III.D). Basis states with
    y >= N are left unchanged.
    """
    _check(n_qubits, x, y)
    _check_modulus(N, y)
    if np.gcd(int(a), N) != 1:
        raise ValueError(f"a={a} and N={N} must be coprime")
    idx, vx = _register_values(n_qubits, x)
    _, vy = _register_values(n_qubits, y)
    powers = np.array([pow(int(a), int(e), N) for e in range(1 << len(x))])
    ok = vy < N
    return _permute(sv, _with_register(n_qubits, idx, y, np.where(ok, (vy * powers[vx]) % N, vy)))
