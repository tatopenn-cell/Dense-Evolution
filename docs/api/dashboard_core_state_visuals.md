# Dashboard Core — State Visuals

Native statevector visualizations — histogram, Q-sphere, per-qubit Bloch
spheres — pure matplotlib/numpy, no Qiskit anywhere.

```python
import numpy as np
from dashboard_core.state_visuals import (
    native_histogram_figure, native_bloch_multivector_figure, native_qsphere_figure,
)

bell = np.array([1, 0, 0, 1]) / np.sqrt(2)  # (|00> + |11>)/sqrt(2)

native_histogram_figure({"00": 512, "11": 488}).savefig("histogram.png")
native_bloch_multivector_figure(bell).savefig("bloch.png")   # both spheres point to the origin -- maximally mixed
native_qsphere_figure(bell).savefig("qsphere.png")            # two equal-size dots at the poles
```

::: dashboard_core.state_visuals

---

**See also**: [`dashboard_core.visuals`](dashboard_core_visuals.md), which
aggregates this with [`circuit_diagram`](dashboard_core_circuit_diagram.md).
