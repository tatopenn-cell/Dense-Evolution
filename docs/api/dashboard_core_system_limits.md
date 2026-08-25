# Dashboard Core — System Limits

Real, per-machine qubit limits — detects actual available RAM instead of
a number picked to fit whatever machine this was developed on, so
Composer refuses an allocation that would actually exhaust memory rather
than crashing partway through.

```python
from dashboard_core.system_limits import max_safe_dense_qubits

limits = max_safe_dense_qubits()
print(limits)
# {'total_mb': 8066.7, 'available_mb': 3039.0, 'threshold_pct': 0.15, 'max_qubits_dense': 27}
# (total_mb/available_mb/max_qubits_dense are real, machine-dependent values --
#  this is one real run's output, not a fixed constant; max_qubits_dense is
#  always between 16 and 27)
```

::: dashboard_core.system_limits
