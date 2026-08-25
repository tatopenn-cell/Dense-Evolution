# Dashboard Core — QM/MM

Real Hellmann-Feynman nuclear forces and a real Velocity-Verlet MD step
for this project's molecule catalog — no fabricated geometry, no
placeholder forces. Backs Composer's QM/MM force and MD-trajectory
panels.

```python
from dashboard_core.qmmm import compute_hellmann_feynman_forces, run_md_trajectory
from dashboard_core.hamiltonians import MOLECULE_CATALOG

# MOLECULE_CATALOG keys are descriptive strings, not bare element symbols:
h2 = [k for k in MOLECULE_CATALOG if k.startswith("H2 ")][0]

forces = compute_hellmann_feynman_forces(h2)
print(forces["energy_hartree"])  # -1.1373 (Hartree-Fock ground state)
print(forces["force_norm"])      # ~0.0154 Hartree/Angstrom -- small residual at equilibrium

trajectory = run_md_trajectory(h2, n_steps=3, dt_fs=0.5)
print(trajectory["force_norm"])  # decreasing as the bond relaxes toward equilibrium
```

::: dashboard_core.qmmm

---

**See also**: [`dashboard_core.hamiltonians`](dashboard_core_hamiltonians.md)
for `MOLECULE_CATALOG` and `build_molecular_hamiltonian`, which this
module's forces are derived from.
