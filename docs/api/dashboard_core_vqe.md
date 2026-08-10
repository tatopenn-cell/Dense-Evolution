# Dashboard Core — VQE

Real, dynamically-generated VQE ansatz circuits for molecular
Hamiltonians — no fixed/hardcoded rotation angles. Every circuit this
module returns is built from the actual molecule's own qubit count and
Hamiltonian, run through [`dense_evolution.autodiff`](autodiff.md)'s
gradient engine.

::: dashboard_core.vqe

---

**See also**: [`dashboard_core.hamiltonians`](dashboard_core_hamiltonians.md)
for where the molecular Hamiltonian this ansatz optimizes against comes
from.
