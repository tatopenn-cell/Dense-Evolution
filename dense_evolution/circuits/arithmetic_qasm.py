"""
Reversible arithmetic as gate circuits, written as OPENQASM 2.0.

`arithmetic.py` applies each operation as a basis permutation
|x> -> |f(x)>: exact, but not a circuit. The functions here return the
gate-level circuits of the original papers as OPENQASM 2.0 text, to be read
with `QASMParser` and run, noised, counted or exported like any other
circuit. Each one is checked exhaustively against the matching permutation
in `arithmetic.py`.

Registers are separate `qreg` declarations, least significant bit first
(`b[0]` is the lowest bit), so the parser's flattened qubit list matches the
register lists `arithmetic.py` takes.

References
----------
Cuccaro, Draper, Kutin, Moulton, quant-ph/0410184 (ripple-carry adder).
Draper, quant-ph/0008033 (addition on a quantum computer, QFT adder).
Beauregard, quant-ph/0205095 (circuit for Shor's algorithm using 2n+3 qubits).
"""
import math

__all__ = [
    "cuccaro_adder_qasm", "draper_adder_qasm",
    "constant_adder_qasm", "modular_constant_adder_qasm",
]

_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

_MAJ = "gate maj c, b, a { cx a, b; cx a, c; ccx c, b, a; }\n"
_UMA = "gate uma c, b, a { ccx c, b, a; cx a, c; cx c, b; }\n"


def _check_n(n):
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")


def _qft(reg, m):
    lines = []
    for k in range(m - 1, -1, -1):
        lines.append(f"h {reg}[{k}];")
        for j in range(k - 1, -1, -1):
            lines.append(f"cp({math.pi / 2 ** (k - j)!r}) {reg}[{j}], {reg}[{k}];")
    return lines


def _iqft(reg, m):
    lines = []
    for k in range(m):
        for j in range(k):
            lines.append(f"cp({-math.pi / 2 ** (k - j)!r}) {reg}[{j}], {reg}[{k}];")
        lines.append(f"h {reg}[{k}];")
    return lines


def _phi_add(reg, m, c, control=None):
    lines = []
    for k in range(m):
        theta = 2 * math.pi * (c % 2 ** (k + 1)) / 2 ** (k + 1)
        if theta == 0.0:
            continue
        if control is None:
            lines.append(f"p({theta!r}) {reg}[{k}];")
        else:
            lines.append(f"cp({theta!r}) {control}, {reg}[{k}];")
    return lines


def cuccaro_adder_qasm(n):
    """
    Ripple-carry adder |a, b, 0> -> |a, a + b, 0> of Cuccaro et al.
    (quant-ph/0410184, Fig. 1-4): MAJ gates carry the sum up, UMA gates
    undo them and write each sum bit.

    Registers: `a[n]`, `b[n+1]` (b[n] receives the carry out, so the sum
    never overflows) and one ancilla `x[1]`, initially 0 and returned to 0.
    2n Toffoli gates and 4n + 1 CNOTs. Matches
    `arithmetic.add_registers(sv, 2n + 2, a, b)`.
    """
    _check_n(n)
    lines = [f"qreg a[{n}];", f"qreg b[{n + 1}];", "qreg x[1];",
             "maj x[0], b[0], a[0];"]
    lines += [f"maj a[{i - 1}], b[{i}], a[{i}];" for i in range(1, n)]
    lines.append(f"cx a[{n - 1}], b[{n}];")
    lines += [f"uma a[{i - 1}], b[{i}], a[{i}];" for i in range(n - 1, 0, -1)]
    lines.append("uma x[0], b[0], a[0];")
    return _HEADER + _MAJ + _UMA + "\n".join(lines) + "\n"


def draper_adder_qasm(n):
    """
    QFT adder |a, b> -> |a, a + b> of Draper (quant-ph/0008033): b is
    taken to Fourier space, each bit of a adds its phase with controlled
    rotations, and the inverse QFT brings the sum back. No ancilla, no
    carry gates.

    Registers: `a[n]`, `b[n+1]` (b[n] holds the carry, so the sum never
    overflows). Matches `arithmetic.add_registers(sv, 2n + 1, a, b)`.
    """
    _check_n(n)
    m = n + 1
    lines = [f"qreg a[{n}];", f"qreg b[{m}];"] + _qft("b", m)
    for i in range(n):
        lines += _phi_add("b", m, 2 ** i, control=f"a[{i}]")
    lines += _iqft("b", m)
    return _HEADER + "\n".join(lines) + "\n"


def constant_adder_qasm(c, n):
    """
    Adds a classical integer, |b> -> |(b + c) mod 2^n>: Beauregard's
    phiADD(c) (quant-ph/0205095, Fig. 3), one phase gate per qubit of b in
    Fourier space, between a QFT and an inverse QFT.

    Register: `b[n]`. Matches `arithmetic.add_constant(sv, n, b, c)`.
    """
    _check_n(n)
    lines = [f"qreg b[{n}];"] + _qft("b", n) + _phi_add("b", n, int(c)) + _iqft("b", n)
    return _HEADER + "\n".join(lines) + "\n"


def modular_constant_adder_qasm(c, N, n):
    """
    Modular addition of a classical integer, |b, 0> -> |(b + c) mod N, 0>
    for 0 <= b < N and 0 <= c < N: Beauregard's phiADD(c)MOD(N)
    (quant-ph/0205095, Fig. 5, here without the two controls), between a
    QFT and an inverse QFT.

    Registers: `b[n+1]` (one bit more than N needs, used as the sign of
    b + c - N) and one ancilla `t[1]`, initially 0 and returned to 0.
    Requires N < 2^n. On inputs with b < N it matches
    `arithmetic.add_constant_mod(sv, n + 2, b[:n], c, N)`; inputs with
    b >= N are outside the circuit's contract.
    """
    _check_n(n)
    c, N = int(c), int(N)
    if not 0 < N < 2 ** n:
        raise ValueError(f"need 0 < N < 2^n = {2 ** n}, got N={N}.")
    if not 0 <= c < N:
        raise ValueError(f"need 0 <= c < N, got c={c}, N={N}.")
    m = n + 1
    msb = f"b[{n}]"
    lines = [f"qreg b[{m}];", "qreg t[1];"] + _qft("b", m)
    lines += _phi_add("b", m, c) + _phi_add("b", m, -N)
    lines += _iqft("b", m) + [f"cx {msb}, t[0];"] + _qft("b", m)
    lines += _phi_add("b", m, N, control="t[0]") + _phi_add("b", m, -c)
    lines += _iqft("b", m) + [f"x {msb};", f"cx {msb}, t[0];", f"x {msb};"] + _qft("b", m)
    lines += _phi_add("b", m, c) + _iqft("b", m)
    return _HEADER + "\n".join(lines) + "\n"
