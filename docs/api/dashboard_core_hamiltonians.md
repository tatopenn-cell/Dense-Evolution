# Dashboard Core — Hamiltonians

Real molecular Hamiltonians, built on demand from actual atomic geometry
via PennyLane's `qchem` module (Hartree-Fock + Jordan-Wigner
fermion-to-qubit mapping) — no fabricated or hand-picked coefficients.
Backs Composer's molecular-energy panel, [`dashboard_core.vqe`](dashboard_core_vqe.md),
and [`dashboard_core.qmmm`](dashboard_core_qmmm.md).

::: dashboard_core.hamiltonians

---

**See also**: [`dashboard_core.vqe`](dashboard_core_vqe.md) for the
ansatz circuits optimized against these Hamiltonians, and
[`dashboard_core.qmmm`](dashboard_core_qmmm.md) for the Hellmann-Feynman
forces derived from them.
