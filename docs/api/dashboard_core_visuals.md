# Dashboard Core — Visuals

Every visualization here is native (plain matplotlib/numpy) — no Qiskit
anywhere in this module. Aggregates
[`circuit_diagram`](dashboard_core_circuit_diagram.md) and
[`state_visuals`](dashboard_core_state_visuals.md) into the set Composer
actually renders.

```python
import numpy as np
from dashboard_core.visuals import draw_circuit_figure, histogram_figure, qsphere_figure, bloch_multivector_figure

bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
draw_circuit_figure([("h", 0), ("cx", 0, 1)], n_qubits=2).savefig("circuit.png")
histogram_figure({"00": 512, "11": 488}).savefig("histogram.png")
qsphere_figure(bell).savefig("qsphere.png")
bloch_multivector_figure(bell).savefig("bloch.png")
```

Same functions and signatures as
[`circuit_diagram`](dashboard_core_circuit_diagram.md)/[`state_visuals`](dashboard_core_state_visuals.md)
directly — this module just pins matplotlib's light style regardless of
any other module's global style changes (e.g. `dense_evolution.registry`
sets a dark background for its own diagnostic plots at import time).

::: dashboard_core.visuals
