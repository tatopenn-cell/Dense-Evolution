# Dashboard Core — Hamiltonians

Real molecular Hamiltonians, built on demand from actual atomic geometry
— no fabricated or hand-picked coefficients. `_get_hamiltonian` dispatches
per molecule: PennyLane's own `qchem` pipeline (Hartree-Fock +
Jordan-Wigner) when every requested element is in PennyLane's bundled
STO-3G table (H2/HeH+/H3+/LiH/H2O), or Dense-Evolution's own
[native Hartree-Fock engine](native_hf.md) otherwise — currently backing
**Si2** (real equilibrium R = 2.184 Å, Balamurugan & Prasad,
[arXiv:cond-mat/0108426](https://arxiv.org/abs/cond-mat/0108426)), whose
elements PennyLane's `dhf` can't reach at all (its table stops at Ne).
Both paths hand off to the same PennyLane `fermionic_observable` +
`jordan_wigner` step, so the qubit mapping is identical either way. Backs
Composer's molecular-energy panel, [`dashboard_core.vqe`](dashboard_core_vqe.md),
and [`dashboard_core.qmmm`](dashboard_core_qmmm.md).

**Honest caveat on Si2**: with only 4 active electrons/orbitals (all 20
core electrons frozen across both atoms), this active space is too small
to reproduce 2.184 Å as its own energy minimum — a direct 10-point
bond-length scan found the minimum at the 1.9 Å edge of the scanned
range, not an interior point. Stated in the catalog entry's own comment
rather than silently picking a geometry that flatters the active-space
choice.

**Ground state without the dense matrix**: `ground_state_energy` needs a
dense `(2**n_qubits, 2**n_qubits)` Hamiltonian, which is exactly what
blocked Si2 in practice (805 MB required, refused by `SafeMemoryGuard` on
modest hardware). `ground_state_energy_sparse` never builds it —
`scipy.sparse.linalg.eigsh` (Lanczos) against a `LinearOperator` wrapping
[`pauli_sum_matvec`](observables.md), matrix-free by construction. Purely
additive: `ground_state_energy`/`build_molecular_hamiltonian` are
unchanged, this is a separate opt-in path for systems too large to
densify at all — verified to match the dense path to ~1e-15 on H2/HeH+,
and to actually succeed on Si2 where the dense path fails.

```python
from dashboard_core.hamiltonians import build_molecular_hamiltonian, ground_state_energy

H, n_qubits = build_molecular_hamiltonian(["H", "H"], [[0, 0, 0], [0, 0, 0.7414]])
print(n_qubits)                    # 4
print(ground_state_energy(H))      # -1.1373 Hartree

# For a catalog molecule too large to densify (e.g. Si2), skip the dense matrix entirely:
from dashboard_core.hamiltonians import ground_state_energy_sparse
e = ground_state_energy_sparse(["H", "H"], [[0, 0, 0], [0, 0, 0.7414]])
print(abs(e - ground_state_energy(H)) < 1e-8)  # True -- matches the dense path exactly
```

::: dashboard_core.hamiltonians

---

**See also**: [`dashboard_core.vqe`](dashboard_core_vqe.md) for the
ansatz circuits optimized against these Hamiltonians,
[`dashboard_core.qmmm`](dashboard_core_qmmm.md) for the Hellmann-Feynman
forces derived from them, and [Native Hartree-Fock](native_hf.md) for the
engine backing Si2 and any other element outside PennyLane's own STO-3G
table.
