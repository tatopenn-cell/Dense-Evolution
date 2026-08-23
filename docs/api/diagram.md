# Diagram (Quirk-style box diagrams)

A Quirk-style matplotlib box-diagram circuit renderer -- reads Dense-Evolution's own
gate-tuple format directly (integers after the gate name are qubit indices, floats are
parameters), so there is no separate gate table to keep in sync by hand. Auto-sized boxes,
red labels for contrast against the cyan/green single-/multi-qubit borders. Deliberately a
different name from [`draw_circuit`](drawing.md) (the plain-ASCII text renderer) rather than
an overload, since the two produce genuinely different output (a saved PNG figure vs. a
printable string) for different purposes.

Promoted from a real reproduction of arXiv:2608.16716's baseband iSWAP pulse
([Dense-Evolution-Discovery Experiment 33](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/germanium_iswap_validation/)),
where it was first used to draw the reference circuit and a single Trotterized pulse slice.

::: dense_evolution.circuits.diagram
