# Dashboard Core — Circuit Diagram

Native circuit-diagram renderer — pure matplotlib, never a Qiskit
`QuantumCircuit`. Replaces `qiskit.circuit.draw(output='mpl')` for
exactly the gate set `dense_evolution` actually supports, so Composer
never depends on Qiskit just to draw a picture.

```python
from dashboard_core.circuit_diagram import draw_native_circuit_diagram

fig = draw_native_circuit_diagram([("h", 0), ("cx", 0, 1)], n_qubits=2)
fig.savefig("bell_pair.png")  # a matplotlib.figure.Figure, save/show it like any other
```

::: dashboard_core.circuit_diagram

---

**See also**: [`dashboard_core.visuals`](dashboard_core_visuals.md), which
aggregates this with [`state_visuals`](dashboard_core_state_visuals.md).
