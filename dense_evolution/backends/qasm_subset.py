"""Reader for the subset of OpenQASM 2.0 the browser simulator accepts.

The parser handles the header, one quantum register, any number of
classical registers, comments, ``barrier`` and ``measure`` (both
ignored), and the gates the NumPy MPS simulator of ``mps_numpy.py``
understands. Several statements can share a line. An angle is an
expression of numbers, ``pi``, ``+ - * /`` and parentheses (nested as
deep as needed), evaluated by a small hand-written parser: ``eval`` and
``exec`` are never used. Every error is a ``ValueError`` whose message
starts with the number of the line. The module imports only ``math`` and
``re``.

References:
    Cross, A. W., Bishop, L. S., Smolin, J. A., Gambetta, J. M. (2017).
        Open Quantum Assembly Language. arXiv:1707.03429. The
        ``OPENQASM 2.0;`` header, the ``qreg``/``creg`` declarations and
        the gate statements are described in section 4 of that paper.
"""

from __future__ import annotations

import math
import re

_1Q_STATIC = ("id", "i", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx")
_1Q_PARAM = ("rx", "ry", "rz", "p", "u1", "phase")
_2Q_STATIC = ("cx", "cy", "cz", "swap")
_2Q_PARAM = ("cp", "crz")

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TOKEN_RE = re.compile(rf"\s*({_NUMBER}|pi|[()+\-*/])")


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
            raise ValueError(f"unexpected token {tok!r} in angle expression") from None
        self._take()
        return num


def _eval_angle(src: str) -> float:
    tokens: list[str] = []
    pos = 0
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ValueError(f"bad character {src[pos:].strip()[0]!r} in the angle {src!r}")
        tokens.append(m.group(1))
        pos = m.end()
    parser = _AngleParser(tokens)
    val = parser.expr()
    if parser._peek() != "":
        raise ValueError(f"trailing tokens in the angle {src!r}")
    return float(val)


def _angle(src: str, lineno: int) -> float:
    try:
        return _eval_angle(src)
    except ValueError as exc:
        raise ValueError(f"line {lineno}: {exc}") from None


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
        ValueError: for a header that is not ``OPENQASM 2.0``, a malformed
            statement, a second ``qreg``, a gate that is not in the
            supported set, a qubit index outside the register, or an angle
            that is not an expression of numbers, ``pi``, ``+ - * /`` and
            parentheses. The message starts with ``line N``.

    Examples:
        >>> nl = chr(10)
        >>> text = nl.join(['OPENQASM 2.0;', 'include "qelib1.inc";',
        ...                  'qreg q[2];', 'creg c[2];', 'h q[0];',
        ...                  'cx q[0],q[1];', 'measure q -> c;']) + nl
        >>> parse_qasm_subset(text)
        (2, [('h', 0), ('cx', 0, 1)])
    """
    n_qubits = 0
    qreg_name = ""
    ops: list = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        code = raw.split("//", 1)[0]
        if not code.strip():
            continue
        *statements, rest = code.split(";")
        if rest.strip():
            raise ValueError(f"line {lineno}: missing ';'")
        for statement in statements:
            body = statement.strip()
            if not body:
                continue
            head = body.split()[0]
            if head == "OPENQASM":
                if body.split() != ["OPENQASM", "2.0"]:
                    raise ValueError(f"line {lineno}: only OPENQASM 2.0 is supported")
            elif head in ("include", "barrier", "measure"):
                continue
            elif head == "qreg":
                if qreg_name != "":
                    raise ValueError(f"line {lineno}: a second qreg is not allowed")
                m = re.match(r"qreg\s+([A-Za-z_]\w*)\[(\d+)\]$", body)
                if not m:
                    raise ValueError(f"line {lineno}: bad qreg declaration")
                qreg_name = m.group(1)
                n_qubits = int(m.group(2))
                if n_qubits < 1:
                    raise ValueError(f"line {lineno}: qreg size must be >= 1")
            elif head == "creg":
                if not re.match(r"creg\s+[A-Za-z_]\w*\[\d+\]$", body):
                    raise ValueError(f"line {lineno}: bad creg declaration")
            else:
                if qreg_name == "":
                    raise ValueError(f"line {lineno}: gate before the qreg declaration")
                ops.append(_parse_gate(body, lineno, n_qubits, qreg_name))
    if qreg_name == "":
        raise ValueError("no qreg in the QASM text")
    return int(n_qubits), ops


def _split_gate(body: str, lineno: int) -> tuple[str, str | None, str]:
    """Split ``name(angle) args`` into its three parts; the angle may nest parentheses."""
    m = re.match(r"([A-Za-z_]\w*)\s*(.*)$", body)
    if not m:
        raise ValueError(f"line {lineno}: cannot parse gate statement")
    name, rest = m.group(1).lower(), m.group(2)
    angle_src = None
    if rest.startswith("("):
        depth = 0
        end = -1
        for pos, char in enumerate(rest):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end < 0:
            raise ValueError(f"line {lineno}: unbalanced parentheses in the angle")
        angle_src = rest[1:end]
        rest = rest[end + 1 :]
    args = rest.strip()
    if not args:
        raise ValueError(f"line {lineno}: cannot parse gate statement")
    return name, angle_src, args


def _parse_gate(body: str, lineno: int, n_qubits: int, qreg_name: str):
    name, angle_src, args = _split_gate(body, lineno)
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
        return (name, qubits[0], _angle(angle_src, lineno))
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
        return (name, qubits[0], qubits[1], _angle(angle_src, lineno))
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
        if idx >= n_qubits:
            raise ValueError(f"line {lineno}: qubit {idx} outside the register q[{n_qubits}]")
        out.append(idx)
    return out
