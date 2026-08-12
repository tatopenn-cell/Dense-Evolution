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

::: dashboard_core.hamiltonians

---

**See also**: [`dashboard_core.vqe`](dashboard_core_vqe.md) for the
ansatz circuits optimized against these Hamiltonians,
[`dashboard_core.qmmm`](dashboard_core_qmmm.md) for the Hellmann-Feynman
forces derived from them, and [Native Hartree-Fock](native_hf.md) for the
engine backing Si2 and any other element outside PennyLane's own STO-3G
table.
