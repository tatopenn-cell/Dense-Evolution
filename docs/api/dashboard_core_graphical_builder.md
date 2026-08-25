# Dashboard Core — Graphical Builder

Turns the ops list produced by the graphical (drag-and-drop) circuit
builder component into `dense_evolution`'s own `(name, *qubits[, param])`
gate-tuple format, so a circuit built visually runs through the exact
same execution path as one written in OpenQASM.

```python
from dashboard_core.graphical_builder import ops_to_native_tuples

# A Bell pair, as the drag-and-drop grid would emit it:
ops = [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]
tuples = ops_to_native_tuples(n_qubits=2, ops=ops)
print(tuples)  # [('h', 0), ('cx', 0, 1)]

# Feed straight into gate_tuples_to_qasm for the QASM the rest of the Composer runs on:
from dashboard_core.qasm_library import gate_tuples_to_qasm
print(gate_tuples_to_qasm(tuples, n_qubits=2))
```

::: dashboard_core.graphical_builder

---

**See also**: [`dashboard_core.circuit_builder_component`](dashboard_core_circuit_builder_component.md),
the UI component this module's output feeds.
