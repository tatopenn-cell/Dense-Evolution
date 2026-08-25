# Dashboard Core — QASM Library

Real, standard OpenQASM 2.0 circuits offered as Composer presets —
textbook examples (Bell pair, GHZ, W-state) and `dense_evolution`'s own
gate library exercised end to end, not synthetic placeholder text.

```python
import dense_evolution as de
from dashboard_core.qasm_library import gate_tuples_to_qasm

# Any hand-written gate-tuple circuit:
print(gate_tuples_to_qasm([('h', 0), ('cx', 0, 1)], n_qubits=2))

# Or any of dense_evolution's own circuit generators, unmodified:
qft_ops = de.qft(3)
print(gate_tuples_to_qasm(qft_ops, n_qubits=3))
```

::: dashboard_core.qasm_library

---

**See also**: [`dense_evolution.parser.QASMParser`](parser.md), which
these presets are meant to be fed into.
