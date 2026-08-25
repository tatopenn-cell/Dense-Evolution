"""
Native circuit-diagram renderer -- pure matplotlib, never a Qiskit
QuantumCircuit. Replaces qiskit's circuit.draw(output='mpl') for exactly
the reason documented in dashboard_core/engine.py's module docstring:
qiskit.circuit.QuantumCircuit.__init__ itself segfaults (SIGSEGV) on
macOS CI runners, on the simplest possible call (QuantumCircuit(3) alone,
no QASM, no methods called on it) -- see tests/integration/test_interop.py::
TestQiskitInterop for the full reproduction story. There is no way to
keep using Qiskit's own drawer without constructing that object, so this
module draws directly from the same (name, *qubits[, param]) gate-tuple
format every other dense_evolution entry point already uses.

Gate vocabulary mirrors dashboard_core/engine.py's dispatch tables
exactly (_ONE_QUBIT_STATIC / _ONE_QUBIT_PARAM / _TWO_QUBIT_STATIC /
_TWO_QUBIT_PARAM / _THREE_QUBIT_STATIC) -- the same gate set QASMParser
can ever hand back, so nothing here can see a name it doesn't recognize.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

__all__ = ['draw_native_circuit_diagram']

_ONE_QUBIT_STATIC = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "id"}
_ONE_QUBIT_PARAM = {"rx", "ry", "rz", "p"}
_TWO_QUBIT_STATIC = {"cx", "cz", "cy", "swap"}
_TWO_QUBIT_PARAM = {"cp", "crz"}
_THREE_QUBIT_STATIC = {"ccx"}

_BOX_LABEL = {
    "h": "H", "x": "X", "y": "Y", "z": "Z", "s": "S", "sdg": "S†",
    "t": "T", "tdg": "T†", "sx": "√X", "id": "I",
    "rx": "RX", "ry": "RY", "rz": "RZ", "p": "P",
    "cy": "Y", "cp": "P", "crz": "RZ",
}

_WIRE_COLOR = "black"
_BOX_FACE = "#6FA8DC"
_BOX_EDGE = "black"
_DOT_COLOR = "black"
_MEASURE_FACE = "#DDDDDD"


def _op_qubits(op, name):
    """Every op tuple is (name, *qubits, *params) -- qubit count is fixed
    per gate name (never variadic), so this just slices the right span."""
    if name in _ONE_QUBIT_STATIC:
        return [op[1]]
    if name in _ONE_QUBIT_PARAM:
        return [op[1]]
    if name in _TWO_QUBIT_STATIC:
        return [op[1], op[2]]
    if name in _TWO_QUBIT_PARAM:
        return [op[1], op[2]]
    if name in _THREE_QUBIT_STATIC:
        return [op[1], op[2], op[3]]
    raise ValueError(f"unsupported gate for native circuit diagram: {name!r}")


def _schedule_columns(ops):
    """Greedy left-packing, same rule every real circuit drawer uses: an
    op goes in the first column strictly after the last column any of its
    qubits was already used in. Returns a list of (op, column) pairs and
    the total column count.

    A multi-qubit op's vertical connector is drawn spanning every row
    between its lowest and highest qubit index (see _draw_vertical_link
    call sites below), not just the qubits it actually touches -- e.g.
    cy(2, 0) draws a line through q1's row even though q1 isn't part of
    the gate. So every qubit in that span, not just the op's own qubits,
    must be marked used at this column too, or an unrelated single-qubit
    gate on the skipped-over qubit can land in the same column and get
    visually cut through by the connector line."""
    next_free_col = {}
    scheduled = []
    for op in ops:
        name = op[0]
        qubits = _op_qubits(op, name)
        span = range(min(qubits), max(qubits) + 1)
        col = max((next_free_col.get(q, 0) for q in span), default=0)
        scheduled.append((op, col))
        for q in span:
            next_free_col[q] = col + 1
    n_columns = max((col for _, col in scheduled), default=-1) + 1
    return scheduled, n_columns


def _draw_one_qubit_box(ax, x, y, label):
    box = Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, facecolor=_BOX_FACE, edgecolor=_BOX_EDGE, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)


def _draw_control_dot(ax, x, y):
    ax.add_patch(Circle((x, y), 0.08, facecolor=_DOT_COLOR, edgecolor=_DOT_COLOR, zorder=4))


def _draw_target_plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.28, facecolor="white", edgecolor=_DOT_COLOR, zorder=3))
    ax.plot([x - 0.28, x + 0.28], [y, y], color=_DOT_COLOR, linewidth=1.4, zorder=4)
    ax.plot([x, x], [y - 0.28, y + 0.28], color=_DOT_COLOR, linewidth=1.4, zorder=4)


def _draw_swap_x(ax, x, y):
    ax.plot([x - 0.15, x + 0.15], [y - 0.15, y + 0.15], color=_DOT_COLOR, linewidth=2, zorder=4)
    ax.plot([x - 0.15, x + 0.15], [y + 0.15, y - 0.15], color=_DOT_COLOR, linewidth=2, zorder=4)


def _draw_vertical_link(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color=_DOT_COLOR, linewidth=1.4, zorder=2)


def _param_label(name, params):
    base = _BOX_LABEL.get(name, name.upper())
    if params:
        return f"{base}({params[0]:.2f})"
    return base


def draw_native_circuit_diagram(ops, n_qubits: int, add_measure: bool = True):
    """Draws a circuit diagram figure directly from dense_evolution gate
    tuples -- no Qiskit QuantumCircuit ever constructed. Qubit 0 is drawn
    at the top, increasing downward, matching Qiskit's own drawer
    convention so this is a drop-in replacement for
    dashboard_core.visuals.draw_circuit_figure's panel.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from dashboard_core.circuit_diagram import draw_native_circuit_diagram
    >>> fig = draw_native_circuit_diagram([('h', 0), ('cx', 0, 1)], n_qubits=2)
    >>> type(fig).__name__
    'Figure'
    >>> fig.savefig('bell_pair.png')  # doctest: +SKIP
    """
    scheduled, n_columns = _schedule_columns(ops)
    n_display_columns = n_columns + (1 if add_measure else 0)

    fig_width = max(3.0, 1.1 * (n_display_columns + 1))
    fig_height = max(2.0, 0.9 * n_qubits + 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def row_y(qubit):
        return n_qubits - 1 - qubit

    wire_x_end = n_display_columns + 0.7
    for q in range(n_qubits):
        y = row_y(q)
        ax.plot([0, wire_x_end], [y, y], color=_WIRE_COLOR, linewidth=1.2, zorder=1)
        ax.text(-0.35, y, f"q{q}", ha="right", va="center", fontsize=10)

    for op, col in scheduled:
        name = op[0]
        x = col + 1.0
        qubits = _op_qubits(op, name)
        rows = [row_y(q) for q in qubits]
        params = list(op[1 + len(qubits):])

        if name in _ONE_QUBIT_STATIC:
            _draw_one_qubit_box(ax, x, rows[0], _BOX_LABEL.get(name, name.upper()))
        elif name in _ONE_QUBIT_PARAM:
            _draw_one_qubit_box(ax, x, rows[0], _param_label(name, params))
        elif name == "swap":
            _draw_vertical_link(ax, x, rows[0], rows[1])
            _draw_swap_x(ax, x, rows[0])
            _draw_swap_x(ax, x, rows[1])
        elif name in _TWO_QUBIT_STATIC:
            # cx/cy/cz: qubits[0] is control, qubits[1] is target.
            _draw_vertical_link(ax, x, rows[0], rows[1])
            _draw_control_dot(ax, x, rows[0])
            if name == "cx":
                _draw_target_plus(ax, x, rows[1])
            else:
                _draw_one_qubit_box(ax, x, rows[1], _BOX_LABEL.get(name, name.upper()))
        elif name in _TWO_QUBIT_PARAM:
            _draw_vertical_link(ax, x, rows[0], rows[1])
            _draw_control_dot(ax, x, rows[0])
            _draw_one_qubit_box(ax, x, rows[1], _param_label(name, params))
        elif name in _THREE_QUBIT_STATIC:
            # ccx (Toffoli): qubits[0], qubits[1] control, qubits[2] target.
            _draw_vertical_link(ax, x, min(rows), max(rows))
            _draw_control_dot(ax, x, rows[0])
            _draw_control_dot(ax, x, rows[1])
            _draw_target_plus(ax, x, rows[2])
        else:
            raise ValueError(f"unsupported gate for native circuit diagram: {name!r}")

    if add_measure:
        x = n_columns + 1.0
        for q in range(n_qubits):
            y = row_y(q)
            box = Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, facecolor=_MEASURE_FACE, edgecolor=_BOX_EDGE, zorder=3)
            ax.add_patch(box)
            ax.text(x, y, "M", ha="center", va="center", fontsize=9, zorder=4)

    ax.set_xlim(-1.2, wire_x_end + 0.3)
    ax.set_ylim(-0.7, n_qubits - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig
