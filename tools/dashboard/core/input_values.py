"""Input values of the qubits of an OpenQASM 2.0 circuit, for the dashboard."""

import re
from collections.abc import Iterable

__all__ = ["qubits_in_qasm", "with_input_values"]

_QREG = re.compile(r"^[ \t]*qreg[ \t]+(\w+)[ \t]*\[[ \t]*(\d+)[ \t]*\][ \t]*;", re.M)
_HEAD = re.compile(r"^[ \t]*(OPENQASM|include|qreg|creg)\b")


def qubits_in_qasm(qasm: str) -> int:
    """Size of the first ``qreg`` declared in an OpenQASM 2.0 text.

    Args:
        qasm: the OpenQASM 2.0 text.

    Returns:
        The number of qubits of the first quantum register, or 0 if the text
        declares none.

    Examples:
        >>> qubits_in_qasm("OPENQASM 2.0;\\nqreg q[3];\\n")
        3
        >>> qubits_in_qasm("OPENQASM 2.0;\\n")
        0
    """
    match = _QREG.search(qasm)
    return int(match.group(2)) if match else 0


def with_input_values(qasm: str, ones: Iterable[int]) -> str:
    """Start the chosen qubits in |1> by putting an ``x`` gate for each one.

    The lines ``x <reg>[i];`` go right after the last declaration line at the
    head of the text (``OPENQASM``, ``include``, ``qreg``, ``creg``), in
    increasing order of ``i``, on the first quantum register.

    Args:
        qasm: the OpenQASM 2.0 text.
        ones: the indices of the qubits that start in |1>. An index that is
            not smaller than the size of the register is ignored.

    Returns:
        The text with the ``x`` lines, or the same text if no index counts.

    Raises:
        ValueError: if an index is negative, or if ``ones`` is not empty and
            the text has no ``qreg`` declaration.

    Examples:
        >>> text = "OPENQASM 2.0;\\nqreg q[2];\\nh q[0];\\n"
        >>> print(with_input_values(text, [1]), end="")
        OPENQASM 2.0;
        qreg q[2];
        x q[1];
        h q[0];
    """
    indices = sorted({int(i) for i in ones})
    for i in indices:
        if i < 0:
            raise ValueError(f"qubit index must be >= 0, got {i}")
    if not indices:
        return qasm
    match = _QREG.search(qasm)
    if match is None:
        raise ValueError("the circuit has no qreg declaration")
    name, size = match.group(1), int(match.group(2))
    indices = [i for i in indices if i < size]
    if not indices:
        return qasm
    lines = qasm.splitlines(keepends=True)
    last = qasm.count("\n", 0, match.start())
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        if not _HEAD.match(line):
            break
        last = max(last, k)
    if not lines[last].endswith("\n"):
        lines[last] += "\n"
    lines[last + 1 : last + 1] = [f"x {name}[{i}];\n" for i in indices]
    return "".join(lines)
