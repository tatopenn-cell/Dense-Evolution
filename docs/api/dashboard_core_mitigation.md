# Dashboard Core — Mitigation (Composer's ZNE panel)

Real Zero-Noise Extrapolation (ZNE) error mitigation, wired to the same
engine and real noise channels ([`dense_evolution.registry.NoiseModel`](registry.md))
the rest of Composer uses — a thin dashboard-facing layer over
[`dense_evolution.mitigation`](mitigation.md), not a separate
reimplementation.

```python
from dashboard_core.mitigation import run_zne_mitigation, run_density_matrix_zne

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
x q[0];
measure q -> c;
"""

result = run_zne_mitigation(qasm, pauli_string="Z", noise_model="bitflip", noise_p=0.05, seed=0)
print(result.ideal_expectation)     # -1.0
print(result.noisy_expectations)    # [-0.880, -0.790, -0.810] -- decays with noise scale
print(result.zne_extrapolated)      # -1.080 -- closer to -1.0 than any single noisy measurement

dm_result = run_density_matrix_zne(qasm, noise_model="bitflip", noise_p=0.05, seed=0)
print(dm_result.fidelity_raw, dm_result.fidelity_corrected)  # 0.940 -> 1.000
```

::: dashboard_core.mitigation

---

**Not to be confused with** [`dense_evolution.mitigation`](mitigation.md)
(same name, different module) — that one is the actual ZNE
implementation (`richardson_extrapolate`, `zne_density_matrix`,
`jsd_predictive_zne_density_matrix`, ...); this one is the dashboard's
request/response wrapper around it.
