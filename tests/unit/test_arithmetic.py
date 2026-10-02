import numpy as np
import pytest

import dense_evolution as de
from dense_evolution.circuits.qft import qft


def _basis(n, i):
    sv = np.zeros(2 ** n, dtype=complex)
    sv[i] = 1.0
    return sv


def _run(n, sv, ops):
    sim = de.DenseSVSimulator(n)
    sim.set_initial_state(sv)
    sim.run_circuit_jit(ops)
    return np.asarray(sim.get_statevector())


def _maj(c, b, a):
    return [('cx', a, b), ('cx', a, c), ('ccx', c, b, a)]


def _uma(c, b, a):
    return [('ccx', c, b, a), ('cx', a, c), ('cx', c, b)]


def _cuccaro(x, a, b, z):
    ops = _maj(x, b[0], a[0])
    for i in range(1, len(a)):
        ops += _maj(a[i - 1], b[i], a[i])
    ops.append(('cx', a[-1], z))
    for i in range(len(a) - 1, 0, -1):
        ops += _uma(a[i - 1], b[i], a[i])
    return ops + _uma(x, b[0], a[0])


def _draper(m, c):
    phases = [('p', m - 1 - j, 2 * np.pi * c * 2 ** j / 2 ** m) for j in range(m)]
    return list(qft(m)) + phases + list(qft(m, inverse=True))


@pytest.mark.parametrize("n_bits", [1, 2, 3])
def test_add_registers_matches_cuccaro_ripple_carry_adder(n_bits):
    n = 2 * n_bits + 2
    x, z = 0, n - 1
    a = list(range(1, 2 * n_bits, 2))
    b = list(range(2, 2 * n_bits + 1, 2))
    ops = _cuccaro(x, a, b, z)
    for i in range(2 ** n):
        if (i >> (n - 1 - x)) & 1:
            continue
        sv = _basis(n, i)
        np.testing.assert_allclose(
            np.asarray(de.add_registers(sv, n, a, b + [z])), _run(n, sv, ops), atol=1e-12)


@pytest.mark.parametrize("c", [1, -1, 3, 6])
def test_add_constant_matches_draper_transform_adder(c):
    m = 3
    for i in range(2 ** m):
        sv = _basis(m, i)
        np.testing.assert_allclose(
            np.asarray(de.add_constant(sv, m, [2, 1, 0], c)), _run(m, sv, _draper(m, c)), atol=1e-10)


def test_addition_modulo_2n_wraps():
    sv = _basis(4, 0b1111)
    out = np.asarray(de.add_registers(sv, 4, [1, 0], [3, 2]))
    assert np.argmax(np.abs(out)) == 0b1110


def test_subtract_is_inverse_of_add():
    rng = np.random.default_rng(0)
    sv = rng.normal(size=32) + 1j * rng.normal(size=32)
    sv /= np.linalg.norm(sv)
    a, b = [1, 0], [4, 3, 2]
    back = de.subtract_registers(de.add_registers(sv, 5, a, b), 5, a, b)
    np.testing.assert_allclose(np.asarray(back), sv, atol=1e-12)


def test_subtract_below_zero_sets_high_bit():
    sv = _basis(5, 0b11010)
    out = np.asarray(de.subtract_registers(sv, 5, [1, 0], [4, 3, 2]))
    assert np.argmax(np.abs(out)) == 0b11111


def test_superposition_is_shifted_not_collapsed():
    sv = np.zeros(8, dtype=complex)
    sv[0] = sv[1] = 1 / np.sqrt(2)
    out = np.asarray(de.add_constant(sv, 3, [2, 1, 0], 3))
    np.testing.assert_allclose(np.abs(out[[3, 4]]), [1 / np.sqrt(2)] * 2, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-12)


def test_shared_qubits_raise():
    with pytest.raises(ValueError, match="share"):
        de.add_registers(_basis(3, 0), 3, [0, 1], [1, 2])


def _bits(n, i, register):
    return sum(((i >> (n - 1 - q)) & 1) << k for k, q in enumerate(register))


@pytest.mark.parametrize("n_bits", [1, 2, 3])
def test_compare_registers_matches_cuccaro_comparator(n_bits):
    n = 2 * n_bits + 2
    x, z = 0, n - 1
    a = list(range(1, 2 * n_bits, 2))
    b = list(range(2, 2 * n_bits + 1, 2))
    ripple = _maj(x, b[0], a[0])
    for i in range(1, n_bits):
        ripple += _maj(a[i - 1], b[i], a[i])
    flips = [('x', q) for q in a]
    ops = flips + ripple + [('cx', a[-1], z)] + ripple[::-1] + flips
    for i in range(2 ** n):
        if (i >> (n - 1 - x)) & 1:
            continue
        sv = _basis(n, i)
        np.testing.assert_allclose(
            np.asarray(de.compare_registers(sv, n, a, b, z, '<')), _run(n, sv, ops), atol=1e-12)


@pytest.mark.parametrize("c", [1, 2, 3])
def test_compare_constant_matches_beauregard_reverse_phiadd(c):
    m, out = 3, 3
    phases = lambda k: [('p', m - 1 - j, 2 * np.pi * k * 2 ** j / 2 ** m) for j in range(m)]
    ops = (list(qft(m)) + phases(-c) + list(qft(m, inverse=True)) + [('cx', 0, out)]
           + list(qft(m)) + phases(c) + list(qft(m, inverse=True)))
    for i in range(8):
        sv = _basis(4, i)
        np.testing.assert_allclose(
            np.asarray(de.compare_constant(sv, 4, [2, 1], c, out, '<')), _run(4, sv, ops), atol=1e-10)


@pytest.mark.parametrize("op, fn", [('<', np.less), ('>', np.greater), ('<=', np.less_equal),
                                    ('>=', np.greater_equal), ('==', np.equal), ('!=', np.not_equal)])
def test_every_comparison_on_every_basis_state(op, fn):
    n, a, b, out = 5, [1, 0], [3, 2], 4
    for i in range(2 ** n):
        got = int(np.argmax(np.abs(np.asarray(de.compare_registers(_basis(n, i), n, a, b, out, op)))))
        assert got == i ^ int(fn(_bits(n, i, a), _bits(n, i, b)))
        got = int(np.argmax(np.abs(np.asarray(de.compare_constant(_basis(n, i), n, b, 2, out, op)))))
        assert got == i ^ int(fn(_bits(n, i, b), 2))


def test_comparison_entangles_a_superposition():
    sv = np.zeros(8, dtype=complex)
    sv[0] = sv[2] = 1 / np.sqrt(2)
    out = np.asarray(de.compare_constant(sv, 3, [1, 0], 1, 2, '<'))
    np.testing.assert_allclose(np.abs(out[[1, 2]]), [1 / np.sqrt(2)] * 2, atol=1e-12)


def test_unknown_comparison_raises():
    with pytest.raises(ValueError, match="op must be"):
        de.compare_registers(_basis(3, 0), 3, [0], [1], 2, '=<')


def _cswap(c, p, q):
    return [('cx', q, p), ('ccx', c, p, q), ('cx', q, p)]


@pytest.mark.parametrize("a", [0, 1, 2])
def test_add_constant_mod_matches_beauregard_modular_adder(a):
    n, b, t, msb, N = 4, [3, 2, 1], 0, 1, 3
    for i in range(16):
        if (i >> 3) & 1 or (i >> 2) & 1 or _bits(n, i, [3, 2]) >= N:
            continue
        v = _basis(n, i)
        v = de.add_constant(v, n, b, a)
        v = de.add_constant(v, n, b, -N)
        v = _run(n, v, [('cx', msb, t)])
        v = de.add_constant_mod(v, n, b, N, 2 ** len(b), controls=[t])
        v = de.add_constant(v, n, b, -a)
        v = _run(n, v, [('x', msb), ('cx', msb, t), ('x', msb)])
        v = de.add_constant(v, n, b, a)
        np.testing.assert_allclose(
            np.asarray(v), np.asarray(de.add_constant_mod(_basis(n, i), n, [3, 2], a, N)), atol=1e-12)


@pytest.mark.parametrize("a", [1, 2, 4])
def test_multiply_add_mod_is_sum_of_controlled_modular_additions(a):
    n, c, x, b, N = 6, 0, [2, 1], [5, 4, 3], 5
    for i in range(2 ** n):
        if _bits(n, i, b) >= N:
            continue
        v = _basis(n, i)
        for k, q in enumerate(x):
            v = de.add_constant_mod(v, n, b, (2 ** k * a) % N, N, controls=[c, q])
        np.testing.assert_allclose(
            np.asarray(v), np.asarray(de.multiply_add_mod(_basis(n, i), n, x, b, a, N, controls=[c])),
            atol=1e-12)


@pytest.mark.parametrize("a", [1, 2])
def test_multiply_mod_matches_beauregard_controlled_ua(a):
    n, c, x, t, N = 5, 0, [2, 1], [4, 3], 3
    for i in range(2 ** n):
        if _bits(n, i, t) or _bits(n, i, x) >= N:
            continue
        v = de.multiply_add_mod(_basis(n, i), n, x, t, a, N, controls=[c])
        v = _run(n, v, sum((_cswap(c, x[k], t[k]) for k in range(2)), []))
        v = de.multiply_add_mod(v, n, x, t, -pow(a, -1, N), N, controls=[c])
        np.testing.assert_allclose(
            np.asarray(v), np.asarray(de.multiply_mod(_basis(n, i), n, x, a, N, controls=[c])), atol=1e-12)


@pytest.mark.parametrize("a", [2, 7, 11])
def test_power_mod_is_chain_of_controlled_multiplications(a):
    n, x, y, N = 7, [2, 1, 0], [6, 5, 4, 3], 15
    for i in range(2 ** n):
        if _bits(n, i, y) >= N:
            continue
        v = _basis(n, i)
        for k, q in enumerate(x):
            v = de.multiply_mod(v, n, y, pow(a, 2 ** k, N), N, controls=[q])
        np.testing.assert_allclose(
            np.asarray(v), np.asarray(de.power_mod(_basis(n, i), n, x, y, a, N)), atol=1e-12)


def test_power_mod_from_one_gives_modular_exponentiation():
    n, x, y, a, N = 7, [2, 1, 0], [6, 5, 4, 3], 7, 15
    for e in range(8):
        i = sum(((e >> k) & 1) << (n - 1 - q) for k, q in enumerate(x)) | (1 << (n - 1 - y[0]))
        out = int(np.argmax(np.abs(np.asarray(de.power_mod(_basis(n, i), n, x, y, a, N)))))
        assert _bits(n, out, y) == pow(a, e, N)
        assert _bits(n, out, x) == e


def test_plain_multiplication_by_odd_number_modulo_2n():
    out = np.asarray(de.multiply_mod(_basis(3, 0b011), 3, [2, 1, 0], 3, 8))
    assert np.argmax(np.abs(out)) == (3 * 3) % 8


def test_modular_maps_are_unitary_on_random_states():
    rng = np.random.default_rng(1)
    sv = rng.normal(size=64) + 1j * rng.normal(size=64)
    sv /= np.linalg.norm(sv)
    for out in (de.add_registers_mod(sv, 6, [1, 0], [3, 2], 3),
                de.multiply_add_mod(sv, 6, [1, 0], [4, 3, 2], 3, 5, controls=[5]),
                de.multiply_mod(sv, 6, [2, 1, 0], 3, 7),
                de.power_mod(sv, 6, [1, 0], [5, 4, 3, 2], 2, 9)):
        np.testing.assert_allclose(np.linalg.norm(np.asarray(out)), 1.0, atol=1e-12)


def test_non_coprime_multiplication_raises():
    with pytest.raises(ValueError, match="coprime"):
        de.multiply_mod(_basis(3, 0), 3, [2, 1, 0], 2, 8)


def test_add_registers_mod_on_every_basis_state():
    n, a, b, N = 4, [1, 0], [3, 2], 3
    for i in range(2 ** n):
        va, vb = _bits(n, i, a), _bits(n, i, b)
        out = int(np.argmax(np.abs(np.asarray(de.add_registers_mod(_basis(n, i), n, a, b, N)))))
        assert _bits(n, out, a) == va
        assert _bits(n, out, b) == ((va + vb) % N if va < N and vb < N else vb)
