"""
Quirk-style box-diagram circuit drawing (matplotlib), promoted from
Dense-Evolution-Discovery Experiment 33's `draw_circuit` utility.

Draws circuits directly in this package's own native gate-tuple format
(the same list DenseSVSimulator.run_circuit accepts) -- integers after the
gate name are qubit indices, floats are parameters, so no gate-name table
needs to be kept in sync with circuits/registry.py's own GATE_IDS.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

__all__ = ["plot_circuit"]


def plot_circuit(circuit, n_qubits, title=None, figsize=None):
    """Draw a circuit in dense_evolution's native tuple format as a
    Quirk-style box diagram.

    circuit : list[tuple]
        E.g. [('x', 1), ('cx', 0, 1), ('iswap', 0, 1), ('rz', 0, 0.5)].
    n_qubits : int

    Returns the matplotlib Figure (not shown/saved automatically).
    """
    WIRE_COLOR = "#4a5266"
    BG = "#0a0a0d"
    TEXT = "#9aa3b2"
    LABEL_COLOR = "#ff5c5c"
    G1_FILL, G1_EDGE = "#0f2a30", "#00e5ff"   # single-qubit gate
    G2_FILL, G2_EDGE = "#0f2a22", "#00ff9d"   # multi-qubit gate
    CHAR_W = 0.145
    MIN_BOX_W, BOX_H, GAP = 0.62, 0.5, 0.18

    def box_width_for(label):
        return max(MIN_BOX_W, len(label) * CHAR_W + 0.22)

    next_free_x = [0.5] * n_qubits
    boxes = []
    for op in circuit:
        name = op[0]
        qubits = [a for a in op[1:] if isinstance(a, int) and not isinstance(a, bool)]
        params = [a for a in op[1:] if isinstance(a, float)]
        if not qubits:
            continue
        label = name.upper() if not params else f"{name.upper()}({params[0]:.2f})"
        w = box_width_for(label)
        x = max(next_free_x[q] for q in qubits) + w / 2
        boxes.append((x, qubits, label, w))
        for q in qubits:
            next_free_x[q] = x + w / 2 + GAP

    total_w = max(next_free_x) + 0.3
    fig_w = figsize[0] if figsize else max(4.0, total_w * 0.9)
    fig_h = figsize[1] if figsize else 0.9 * n_qubits + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot([0, total_w], [y, y], color=WIRE_COLOR, lw=1.5, zorder=1)
        ax.text(-0.15, y, f"q{q}", ha="right", va="center", color=TEXT,
                 fontsize=11, family="monospace")

    for x, qubits, label, box_w in boxes:
        ys = [n_qubits - 1 - q for q in qubits]
        y_lo, y_hi = min(ys), max(ys)
        is_multi = len(qubits) > 1
        fill, edge = (G2_FILL, G2_EDGE) if is_multi else (G1_FILL, G1_EDGE)
        if is_multi:
            ax.plot([x, x], [y_lo, y_hi], color=edge, lw=1.3, zorder=2)
            for y in ys:
                ax.add_patch(Circle((x, y), 0.045, color=edge, zorder=4))
            rect = Rectangle((x - box_w / 2, y_lo - BOX_H / 2),
                              box_w, (y_hi - y_lo) + BOX_H,
                              facecolor=fill, edgecolor=edge, lw=1.4, zorder=3)
            ax.add_patch(rect)
            ax.text(x, (y_lo + y_hi) / 2, label, ha="center", va="center",
                     color=LABEL_COLOR, fontsize=10.5, family="monospace", fontweight="bold", zorder=5)
        else:
            y = ys[0]
            rect = Rectangle((x - box_w / 2, y - BOX_H / 2), box_w, BOX_H,
                              facecolor=fill, edgecolor=edge, lw=1.4, zorder=3)
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", color=LABEL_COLOR,
                     fontsize=10.5, family="monospace", fontweight="bold", zorder=4)

    ax.set_xlim(-0.6, total_w)
    ax.set_ylim(-0.6, n_qubits - 0.4)
    ax.axis("off")
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, family="monospace", pad=12)
    plt.tight_layout()
    return fig
