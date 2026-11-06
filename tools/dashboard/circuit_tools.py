"""Helpers of the Build page of the dashboard: input values, result tables, topology.

This file sits next to ``app.py`` on purpose: the installers of the dashboard
download ``app.py`` and the files beside it from GitHub, and take
``dashboard_core`` from the released package, so code that the new ``app.py``
needs cannot live in ``dashboard_core`` until the next release.
"""

import random
import re
from collections.abc import Iterable, Sequence
from typing import Any

from dashboard_core.qasm_library import gate_tuples_to_qasm

import dense_evolution as de
from dense_evolution.circuits.topology import VALID_PATTERNS

__all__ = [
    "min_qubits",
    "with_qubit_count",
    "measurement_rows",
    "outcome_rows",
    "qiskit_code",
    "qubits_in_qasm",
    "state_csv",
    "topology_qasm",
    "with_input_values",
]

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


def measurement_rows(
    counts: dict[str, int],
    input_ones: Iterable[int],
    n_qubits: int,
    seed: int = 0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """The shots of a run one by one: measurement number, input and output.

    The run keeps only how many times each outcome came out, so the order of the
    shots is drawn again from those counts, with ``seed``; every count is kept.

    Args:
        counts: shot count of each outcome, as a bitstring.
        input_ones: the qubits that start in |1>.
        n_qubits: size of the register.
        seed: seed of the order of the shots.
        limit: how many measurements to return at most.

    Returns:
        ``[{"measurement": 1, "input": "00", "output": "11"}, ...]``, the qubit 0 being
        the rightmost bit in both strings, as in the outcomes of a run.

    Examples:
        >>> rows = measurement_rows({"11": 2, "00": 1}, [], 2, seed=1)
        >>> sorted(r["output"] for r in rows)
        ['00', '11', '11']
        >>> rows[0]["input"]
        '00'
    """
    ones = set(input_ones)
    start = "".join("1" if q in ones else "0" for q in reversed(range(n_qubits)))
    shots = [bits for bits, n in sorted(counts.items()) for _ in range(int(n))]
    random.Random(seed).shuffle(shots)
    return [
        {"measurement": k + 1, "input": start, "output": bits}
        for k, bits in enumerate(shots[:limit])
    ]


_SIMPLE = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "id"}
_ANGLE_ONE = {"rx": "rx", "ry": "ry", "rz": "rz", "p": "p", "u1": "p", "phase": "p"}
_TWO = {"cx", "cy", "cz", "swap"}
_ANGLE_TWO = {"cp", "crz"}


def qiskit_code(ops: Sequence[tuple], n_qubits: int) -> str:
    """Qiskit program of a circuit given as gate tuples.

    Args:
        ops: dense_evolution gate tuples, ``("h", 0)``, ``("cp", 0, 1, 0.5)``, ...
        n_qubits: size of the register.

    Returns:
        Python code that builds the same circuit as a Qiskit ``QuantumCircuit``,
        with a measurement of every qubit at the end.

    Raises:
        ValueError: a gate has no Qiskit method of the same name.

    Examples:
        >>> print(qiskit_code([("h", 0), ("cx", 0, 1)], 2).splitlines()[3])
        qc.h(0)
    """
    lines = [
        "from qiskit import QuantumCircuit",
        "",
        f"qc = QuantumCircuit({n_qubits}, {n_qubits})",
    ]
    for op in ops:
        name, args = op[0], op[1:]
        if name in _SIMPLE or name in _TWO or name == "ccx":
            lines.append(f"qc.{name}({', '.join(str(a) for a in args)})")
        elif name in _ANGLE_ONE:
            lines.append(f"qc.{_ANGLE_ONE[name]}({args[1]!r}, {args[0]})")
        elif name in _ANGLE_TWO:
            lines.append(f"qc.{name}({args[2]!r}, {args[0]}, {args[1]})")
        else:
            raise ValueError(f"no Qiskit method for the gate {name!r}")
    lines.append(f"qc.measure(range({n_qubits}), range({n_qubits}))")
    return "\n".join(lines) + "\n"


def state_csv(statevector: Sequence[complex], n_qubits: int) -> str:
    """The final state as a CSV text: one row for each basis state.

    Args:
        statevector: amplitudes in the Qiskit order (qubit 0 is the rightmost bit).
        n_qubits: size of the register.

    Returns:
        A CSV with the columns ``state,re,im,probability``.

    Examples:
        >>> state_csv([2 ** -0.5, 0, 0, 2 ** -0.5], 2).splitlines()[0]
        'state,re,im,probability'
    """
    rows = ["state,re,im,probability"]
    for i, amp in enumerate(statevector):
        amp = complex(amp)
        rows.append(f"{format(i, f'0{n_qubits}b')},{amp.real!r},{amp.imag!r},{abs(amp) ** 2!r}")
    return "\n".join(rows) + "\n"


def min_qubits(qasm: str) -> int:
    """Fewest qubits a circuit can have: the highest qubit its gates use, plus one.

    Args:
        qasm: the OpenQASM 2.0 text.

    Returns:
        The highest index written in a gate line plus one, at least 1.

    Examples:
        >>> min_qubits('qreg q[5];\\ncreg c[5];\\nh q[0];\\ncx q[0],q[2];\\n')
        3
        >>> min_qubits('qreg q[4];\\n')
        1
    """
    used = [
        int(i)
        for line in qasm.splitlines()
        if not _HEAD.match(line)
        for i in re.findall(r"\[\s*(\d+)\s*\]", line)
    ]
    return max(used) + 1 if used else 1


def dense_qubit_cap(
    available_mb: float,
    total_mb: float,
    threshold: float = 0.15,
    negligible_mb: float = 512.0,
    ceiling: int = 30,
) -> int:
    """Largest register a dense statevector can have on this machine.

    The rule is the one of ``SafeMemoryGuard.check_allocation``: after allocating the
    ``2 ** n`` complex128 amplitudes, ``threshold`` of the total RAM must still be free;
    an allocation under ``negligible_mb`` is always allowed while that much RAM is free.

    Args:
        available_mb: free RAM now, in MB.
        total_mb: total RAM, in MB.
        threshold: share of the total RAM that must stay free (the guard uses 15%).
        negligible_mb: size below which the guard does not check.
        ceiling: the largest size the dense simulator is meant for.

    Returns:
        The largest qubit count that passes, at least 1.

    Examples:
        >>> dense_qubit_cap(8000, 16000)
        28
        >>> dense_qubit_cap(300, 300)
        23
    """
    n = 1
    while n < ceiling:
        need = 2 ** (n + 1) * 16 / 1e6
        if need < negligible_mb and available_mb > negligible_mb:
            fits = True
        else:
            left = available_mb - need
            fits = left >= 0 and left / total_mb >= threshold
        if not fits:
            break
        n += 1
    return n


def with_qubit_count(qasm: str, n_qubits: int) -> str:
    """The same circuit with the first ``qreg`` and the first ``creg`` set to ``n_qubits``.

    Args:
        qasm: the OpenQASM 2.0 text.
        n_qubits: the new size of the registers.

    Returns:
        The text with both sizes changed; the gates are not touched.

    Examples:
        >>> with_qubit_count('qreg q[2];\\ncreg c[2];\\nh q[0];\\n', 4)
        'qreg q[4];\\ncreg c[4];\\nh q[0];\\n'
    """
    for reg in ("qreg", "creg"):
        qasm = re.sub(
            rf"({reg}[ \t]+\w+[ \t]*\[)[ \t]*\d+([ \t]*\])", rf"\g<1>{n_qubits}\g<2>", qasm, count=1
        )
    return qasm
