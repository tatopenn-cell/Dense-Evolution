# Dashboard Core — VQE

Real, dynamically-generated VQE ansatz circuits for molecular
Hamiltonians — no fixed/hardcoded rotation angles. Every circuit this
module returns is built from the actual molecule's own qubit count and
Hamiltonian, run through [`dense_evolution.autodiff`](autodiff.md)'s
gradient engine.

## Quick start

```bash
pip install dense-evolution[pennylane]
```

```python
from dashboard_core.vqe import run_vqe

result = run_vqe(
    symbols=["H", "H"],
    geometry=[[0, 0, 0], [0, 0, 0.7414]],
    ansatz_type="hardware_efficient",
    n_layers=4,
    maxiter=200,
)
print(result["vqe_energy_hartree"])  # -1.137270 (verified against exact_energy_hartree)
```

`run_vqe` needs the optional `pennylane` extra (used internally to build
the molecular Hamiltonian and Hartree-Fock reference state — the native
UCCSD ansatz circuits themselves don't need PennyLane, see
[`dense_evolution.circuits.uccsd`](../api/index.md), but Hamiltonian
construction still does). `ansatz_type="uccsd"` uses that native
implementation instead of `"hardware_efficient"`'s generic layered
ansatz; `n_layers` is then ignored, since UCCSD's parameter count comes
from the molecule's own occupied/virtual orbital structure.

::: dashboard_core.vqe

---

**See also**: [`dashboard_core.hamiltonians`](dashboard_core_hamiltonians.md)
for where the molecular Hamiltonian this ansatz optimizes against comes
from.
