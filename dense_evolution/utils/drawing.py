"""
Plain-text circuit diagrams -- the fast way to sanity-check what a
gate-tuple list actually builds without running it, the same job
Qiskit's `circuit.draw()` or Cirq's text diagrams do.

Uses plain ASCII ('-', '|', '*') rather than Unicode box-drawing
characters deliberately: a diagram meant to be printed to a terminal or
written to a log file needs to survive whatever console encoding the
caller happens to have (this was tested against, and would otherwise
crash on, a default Windows cp1252 console).
"""

__all__ = ['draw_circuit']

_LABELS = {
    'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z', 's': 'S', 'sdg': 'Sdg',
    't': 'T', 'tdg': 'Tdg', 'sx': 'Sx', 'id': 'I',
    'rx': 'Rx', 'ry': 'Ry', 'rz': 'Rz', 'p': 'P', 'u1': 'P', 'phase': 'P',
}
_TWO_QUBIT_TARGET_SYMBOL = {'cx': 'X', 'cz': 'Z', 'cy': 'Y', 'cp': 'P', 'crz': 'Rz'}


def draw_circuit(circuit, n_qubits):
    """
    Render a circuit (in the gate-tuple form run_circuit accepts) as a
    plain-text diagram, one gate per column, one line per qubit.

    Parameters
    ----------
    circuit : list[tuple]
        Gate ops, as passed to `DenseSVSimulator.run_circuit`. Supports
        every gate this package's transpiler recognizes: single-qubit
        static and parametric gates, cx/cz/cy/cp/crz, swap, and ccx.
    n_qubits : int
        Number of qubit rows to draw, must be >= 1 (should cover every
        qubit index referenced in `circuit`).

    Returns
    -------
    str
        Multi-line diagram, one row per qubit (``q0: --H----*--``, ...).
        Print it, or split on '\\n' to inspect individual rows.

    Examples
    --------
    >>> import dense_evolution as de
    >>> print(de.draw_circuit([('h', 0), ('cx', 0, 1)], n_qubits=2))
    q0: --H----*--
    q1: -------X--
    """
    if n_qubits < 1:
        raise ValueError(f"draw_circuit needs at least 1 qubit, got {n_qubits}")

    prefixes = [f"q{q}: " for q in range(n_qubits)]
    prefix_width = max(len(p) for p in prefixes)
    rows = [p.ljust(prefix_width) for p in prefixes]

    for op in circuit:
        name = op[0]
        col = {}  # qubit -> symbol to draw in this column (None = plain wire)

        if name == 'swap':
            a, b = op[1], op[2]
            lo, hi = min(a, b), max(a, b)
            for q in range(n_qubits):
                col[q] = 'x' if q in (a, b) else ('|' if lo < q < hi else None)
        elif name == 'ccx':
            c1, c2, tgt = op[1], op[2], op[3]
            lo, hi = min(c1, c2, tgt), max(c1, c2, tgt)
            for q in range(n_qubits):
                if q in (c1, c2):
                    col[q] = '*'
                elif q == tgt:
                    col[q] = 'X'
                else:
                    col[q] = '|' if lo < q < hi else None
        elif name in _TWO_QUBIT_TARGET_SYMBOL:
            ctrl, tgt = op[1], op[2]
            lo, hi = min(ctrl, tgt), max(ctrl, tgt)
            for q in range(n_qubits):
                if q == ctrl:
                    col[q] = '*'
                elif q == tgt:
                    col[q] = _TWO_QUBIT_TARGET_SYMBOL[name]
                else:
                    col[q] = '|' if lo < q < hi else None
        else:
            q = op[1]
            col[q] = _LABELS.get(name, name.upper())

        label_width = max((len(v) for v in col.values() if v is not None), default=1)
        seg_width = label_width + 4  # 2 filler chars on each side

        for q in range(n_qubits):
            content = col.get(q)
            if content is None:
                rows[q] += '-' * seg_width
            else:
                rows[q] += '--' + content.center(label_width) + '--'

    return '\n'.join(rows)
