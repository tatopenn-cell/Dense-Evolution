# Dashboard Core — Engine

The real simulation engine for the dashboard: runs an actual
[`DenseSVSimulator`](simulator.md) circuit, not a mocked/placeholder
result, and is what Composer's UI and the MCP server's simulation tools
both call underneath.

```python
from dashboard_core.engine import run_circuit_from_qasm

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

result = run_circuit_from_qasm(qasm, n_shots=1000, seed=0)
print(result.probabilities.round(3))  # [0.5 0.  0.  0.5] -- Bell pair
print(result.counts)                  # e.g. {'00': 473, '11': 527}
```

For circuits too large for a dense statevector, `run_large_circuit_mps` finds the
top-*k* most probable basis states via MPS instead:

```python
from dashboard_core.engine import run_large_circuit_mps

result = run_large_circuit_mps(qasm, k=4, seed=0)
print(result.top_k_states[:2])  # [('00', 0.5), ('11', 0.5)] -- the rest are ~0
```

::: dashboard_core.engine
