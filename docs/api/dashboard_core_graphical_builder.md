# Dashboard Core — Graphical Builder

Turns the ops list produced by the graphical (drag-and-drop) circuit
builder component into `dense_evolution`'s own `(name, *qubits[, param])`
gate-tuple format, so a circuit built visually runs through the exact
same execution path as one written in OpenQASM.

::: dashboard_core.graphical_builder

---

**See also**: [`dashboard_core.circuit_builder_component`](dashboard_core_circuit_builder_component.md),
the UI component this module's output feeds.
