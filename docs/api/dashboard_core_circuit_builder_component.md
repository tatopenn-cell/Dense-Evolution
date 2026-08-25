# Dashboard Core — Circuit Builder Component

Graphical (drag-and-drop) circuit builder — the fourth pillar of IBM
Quantum Composer's layout (graphical editor + code editor + statevector
view + results) reproduced natively for this project's own dashboard.

This is a real Streamlit component (`st.components.v2.component`) — it only means
something rendered inside a running Streamlit app (`streamlit run app.py`), unlike
every other `dashboard_core` module, which is a plain importable function you can
call from a script or a REPL. There is nothing to execute standalone here; the
snippet below is the exact call `tools/dashboard/app.py` makes, reproduced as a
minimal `streamlit run`-able app:

```python
import streamlit as st
import dashboard_core as dc

n_qubits = st.number_input("Qubits", min_value=1, max_value=8, value=3, step=1)
ops = dc.mount_circuit_builder(int(n_qubits), n_columns=12, key=f"circuit_builder_{int(n_qubits)}")

if ops:
    st.write(ops)  # [{'gate': 'h', 'qubits': [0]}, ...] as gates are dropped onto the grid
    tuples = dc.ops_to_native_tuples(int(n_qubits), ops)
    st.code(dc.gate_tuples_to_qasm(tuples, int(n_qubits)))
```

`key` must change whenever `n_qubits` changes (include it in the key, as above), or
Streamlit reuses a stale grid sized for a different qubit count.

::: dashboard_core.circuit_builder_component

---

**See also**: [`dashboard_core.graphical_builder`](dashboard_core_graphical_builder.md),
which converts this component's output into runnable gate tuples.
