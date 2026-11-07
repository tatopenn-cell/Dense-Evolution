"""Reader for the subset of OpenQASM 2.0 the browser simulator accepts.

The parser handles the header, one quantum register, any number of
classical registers, comments, ``barrier`` and ``measure`` (both
ignored), and the gates the NumPy MPS simulator of ``mps_numpy.py``
understands. An angle is an expression of numbers, ``pi``, ``+ - * /``
and parentheses, evaluated by a small hand-written parser: ``eval`` and
``exec`` are never used. The module imports only ``math`` and ``re``.

References:
    Cross, A. W., Bishop, L. S., Smolin, J. A., Gambetta, J. M. (2017).
        Open Quantum Assembly Language. arXiv:1707.03429. The
        ``OPENQASM 2.0;`` header, the ``qreg``/``creg`` declarations and
        the gate statements are described in section 4 of that paper.
"""

import math
import re

_1Q_STATIC = ("id", "i", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx")
_1Q_PARAM = ("rx", "ry", "rz", "p", "u1", "phase")
_2Q_STATIC = ("cx", "cy", "cz", "swap")
_2Q_PARAM = ("cp", "crz")

_TOKEN_RE = re.compile(r"\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|pi|[()+\-*/])")


class _AngleParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def _take(self) -> str:
        tok = self._peek()
        if tok == "":
            raise ValueError("unexpected end of angle expression")
        self.pos += 1
        return tok

    def expr(self) -> float:
        val = self.term()
        while self._peek() in ("+", "-"):
            op = self._take()
            rhs = self.term()
            val = val + rhs if op == "+" else val - rhs
        return val

    def term(self) -> float:
        val = self.factor()
        while self._peek() in ("*", "/"):
            op = self._take()
            rhs = self.factor()
            if op == "*":
                val = val * rhs
            else:
                if rhs == 0.0:
                    raise ValueError("division by zero in angle expression")
                val = val / rhs
        return val

    def factor(self) -> float:
        tok = self._peek()
        if tok == "":
            raise ValueError("unexpected end of angle expression")
        if tok == "-":
            self._take()
            return -self.factor()
        if tok == "+":
            self._take()
            return self.factor()
        if tok == "(":
            self._take()
            val = self.expr()
            if self._take() != ")":
                raise ValueError("expected ')' in angle expression")
            return val
        if tok == "pi":
            self._take()
            return math.pi
        try:
            num = float(tok)
        except ValueError:
            raise ValueError(f"unexpected token {tok!r} in angle expression")
        self._take()
        return num


def _eval_angle(src: str) -> float:
    tokens: list[str] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ValueError(f"angle: bad character at position {pos}")
        tokens.append(m.group(1))
        pos = m.end()
    parser = _AngleParser(tokens)
    val = parser.expr()
    if parser._peek() != "":
        raise ValueError(f"angle: trailing tokens in {src!r}")
    return float(val)


def parse_qasm_subset(text: str) -> tuple[int, list]:
    """Parse the OpenQASM subset and return ``(n_qubits, ops)``.

    Args:
        text: the QASM text.

    Returns:
        ``(n_qubits, ops)``, with ``ops`` the gate tuples of the
        simulator: ``("h", q)``, ``("rx", q, angle)``,
        ``("cx", q1, q2)``, ``("cp", q1, q2, angle)``,
        ``("ccx", c1, c2, tgt)``.

    Raises:
        ValueError: for a malformed statement, a second ``qreg``, a gate
            that is not in the supported set, a qubit index outside the
            register, or an angle that is not an expression of numbers,
            ``pi``, ``+ - * /`` and parentheses. The message names the
            line.

    Examples:
        >>> nl = chr(10)
        >>> text = nl.join(['OPENQASM 2.0;', 'include "qelib1.inc";',
        ...                  'qreg q[2];', 'creg c[2];', 'h q[0];',
        ...                  'cx q[0],q[1];', 'measure q -> c;']) + nl
        >>> parse_qasm_subset(text)
        (2, [('h', 0), ('cx', 0, 1)])
    """
    lines = text.splitlines()
    n_qubits = 0
    qreg_name = ""
    ops = []
    for lineno, raw in enumerate(lines, 1):
        stmt = raw.split("//", 1)[0].strip()
        if not stmt:
            continue
        if not stmt.endswith(";"):
            raise ValueError(f"line {lineno}: missing ';'")
        body = stmt[:-1].strip()
        head = body.split()[0] if body else ""
        if head == "OPENQASM":
            continue
        if head == "include":
            continue
        if head == "qreg":
            m = re.match(r"qreg\s+([A-Za-z_]\w*)\[(\d+)\]$", body)
            if not m:
                raise ValueError(f"line {lineno}: bad qreg declaration")
            if qreg_name != "":
                raise ValueError(f"line {lineno}: a second qreg is not allowed")
            qreg_name = m.group(1)
            n_qubits = int(m.group(2))
            if n_qubits < 1:
                raise ValueError(f"line {lineno}: qreg size must be >= 1")
            continue
        if head == "creg":
            if not re.match(r"creg\s+[A-Za-z_]\w*\[\d+\]$", body):
                raise ValueError(f"line {lineno}: bad creg declaration")
            continue
        if head == "barrier":
            continue
        if head == "measure":
            continue
        if qreg_name == "":
            raise ValueError(f"line {lineno}: gate before the qreg declaration")
        ops.append(_parse_gate(body, lineno, n_qubits, qreg_name))
    if qreg_name == "":
        raise ValueError("no qreg in the QASM text")
    return int(n_qubits), ops


def _parse_gate(body: str, lineno: int, n_qubits: int, qreg_name: str):
    m = re.match(r"([A-Za-z_]\w*)(?:\(([^)]*)\))?\s+(.+)$", body)
    if not m:
        raise ValueError(f"line {lineno}: cannot parse gate statement")
    name = m.group(1).lower()
    angle_src = m.group(2)
    args = m.group(3)
    qubits = _parse_qubit_args(args, lineno, qreg_name, n_qubits)
    if name in _1Q_STATIC:
        if angle_src is not None:
            raise ValueError(f"line {lineno}: {name} takes no angle")
        if len(qubits) != 1:
            raise ValueError(f"line {lineno}: {name} takes 1 qubit")
        return (name, qubits[0])
    if name in _1Q_PARAM:
        if angle_src is None:
            raise ValueError(f"line {lineno}: {name} needs an angle")
        if len(qubits) != 1:
            raise ValueError(f"line {lineno}: {name} takes 1 qubit")
        return (name, qubits[0], _eval_angle(angle_src))
    if name in _2Q_STATIC:
        if angle_src is not None:
            raise ValueError(f"line {lineno}: {name} takes no angle")
        if len(qubits) != 2:
            raise ValueError(f"line {lineno}: {name} takes 2 qubits")
        return (name, qubits[0], qubits[1])
    if name in _2Q_PARAM:
        if angle_src is None:
            raise ValueError(f"line {lineno}: {name} needs an angle")
        if len(qubits) != 2:
            raise ValueError(f"line {lineno}: {name} takes 2 qubits")
        return (name, qubits[0], qubits[1], _eval_angle(angle_src))
    if name == "ccx":
        if angle_src is not None:
            raise ValueError(f"line {lineno}: ccx takes no angle")
        if len(qubits) != 3:
            raise ValueError(f"line {lineno}: ccx takes 3 qubits")
        return (name, qubits[0], qubits[1], qubits[2])
    raise ValueError(f"line {lineno}: unknown gate {name!r}")


def _parse_qubit_args(args: str, lineno: int, qreg_name: str, n_qubits: int) -> list[int]:
    out: list[int] = []
    pat = re.compile(rf"{re.escape(qreg_name)}\[(\d+)\]$")
    for part in args.split(","):
        part = part.strip()
        m = pat.match(part)
        if not m:
            raise ValueError(f"line {lineno}: bad qubit argument {part!r}")
        idx = int(m.group(1))
        if idx < 0 or idx >= n_qubits:
            raise ValueError(f"line {lineno}: qubit {idx} outside the register q[{n_qubits}]")
        out.append(idx)
    return out
