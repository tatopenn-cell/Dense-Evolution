"""Helpers of the Build page of the dashboard: input values, result tables, topology.

This file sits next to ``app.py`` on purpose: the installers of the dashboard
download ``app.py`` and the files beside it from GitHub, and take
``dashboard_core`` from the released package, so code that the new ``app.py``
needs cannot live in ``dashboard_core`` until the next release.
"""

import re
from collections.abc import Iterable
from typing import Any

import dense_evolution as de
from dashboard_core.qasm_library import gate_tuples_to_qasm
from dense_evolution.circuits.topology import VALID_PATTERNS

__all__ = ["outcome_rows", "qubits_in_qasm", "topology_qasm", "with_input_values"]

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


def outcome_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Rows of the outcomes of a run, most frequent first.

    Args:
        counts: shot count of each outcome, as a bitstring.

    Returns:
        A list of ``{"outcome": bitstring, "shots": n, "probability": n / total}``
        sorted by ``shots`` descending and, for equal ``shots``, by ``outcome``.
        An empty list if ``counts`` is empty or has no shot.

    Examples:
        >>> outcome_rows({"11": 497, "00": 503})[0]
        {'outcome': '00', 'shots': 503, 'probability': 0.503}
    """
    total = sum(counts.values())
    if total <= 0:
        return []
    rows: list[dict[str, Any]] = [
        {"outcome": key, "shots": int(n), "probability": n / total} for key, n in counts.items()
    ]
    rows.sort(key=lambda row: (-row["shots"], row["outcome"]))
    return rows


_GATES = ("cx", "cz", "cy")


def topology_qasm(
    n_qubits: int,
    pattern: str,
    gate: str = "cx",
    reverse: bool = False,
    hub: int = 0,
    hadamards: bool = True,
) -> str:
    """OpenQASM 2.0 of an entangling layer built by ``entangling_layer``.

    Args:
        n_qubits: number of qubits, at least 2.
        pattern: one of ``dense_evolution.circuits.topology.VALID_PATTERNS``.
        gate: two-qubit gate of every edge: ``"cx"``, ``"cz"`` or ``"cy"``.
        reverse: swap control and target of every edge.
        hub: hub qubit, used only by the ``"star"`` pattern.
        hadamards: start with one Hadamard on every qubit.

    Returns:
        The OpenQASM text of the circuit.

    Raises:
        ValueError: for fewer than 2 qubits, an unknown pattern, an unknown
            gate, or a ``hub`` outside ``0 <= hub < n_qubits`` with ``"star"``.

    Examples:
        >>> print(topology_qasm(3, "linear", hadamards=False), end="")
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        cx q[0],q[1];
        cx q[1],q[2];
        measure q -> c;
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"pattern must be one of {VALID_PATTERNS}, got {pattern!r}")
    if gate not in _GATES:
        raise ValueError(f"gate must be one of {_GATES}, got {gate!r}")
    if pattern == "star" and not 0 <= hub < n_qubits:
        raise ValueError(f"hub must satisfy 0 <= hub < {n_qubits}, got {hub}")
    ops = de.entangling_layer(n_qubits, pattern=pattern, gate=gate, reverse=reverse, hub=hub)
    if hadamards:
        ops = [("h", q) for q in range(n_qubits)] + ops
    return gate_tuples_to_qasm(ops, n_qubits)
