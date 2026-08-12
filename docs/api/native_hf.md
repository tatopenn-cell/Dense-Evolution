# Native Hartree-Fock (`dense_evolution.native_hf`)

A from-scratch, JAX-vectorized ab-initio Hartree-Fock engine — overlap,
kinetic, nuclear-attraction, and electron-repulsion integrals over s/p
Cartesian Gaussian shells via the Obara-Saika recursion (Obara & Saika,
*J. Chem. Phys.* 84, 3963, 1986), each shell-pair/quartet batched with
`jax.lax.scan`/`jax.vmap` and compiled with `jax.jit` instead of looping
in the Python interpreter.

It exists because PennyLane's own differentiable Hartree-Fock solver
(`qml.qchem`, `method="dhf"`) builds these same integrals through a
Python-level loop wrapped in its autograd-tracing numpy layer — correct,
but profiled directly at 482 of 483 total seconds for Si2/STO-3G, almost
entirely per-scalar-op tracer overhead rather than real FLOPs. This
module only replaces the integral/SCF stage; the converged result still
goes to PennyLane's own `fermionic_observable` + `jordan_wigner` for the
qubit mapping, since that stage is already fast (under 2 seconds) and
well-tested. Basis-set parameters come from the
[`basis_set_exchange`](https://github.com/MolSSI-BSE/basis_set_exchange)
package, so any element it has STO-3G data for is reachable, not just
PennyLane's bundled H–Ne table. An element needing d-orbitals or higher
(e.g. Fe) fails with a clear `NotImplementedError` naming the real
limitation, not a silent wrong energy for an incomplete basis.

Design and algorithm structure were informed by studying
[lowdanie/hartree-fock-solver](https://github.com/lowdanie/hartree-fock-solver)
("slaterform", Apache-2.0) as a reference for structuring the Obara-Saika
recursion with `jax.lax.scan`, and by PennyLane's own white paper
(Delgado et al., "Differentiable quantum computational chemistry with
PennyLane", [arXiv:2111.09967](https://arxiv.org/abs/2111.09967)) — no
source code from either project is copied here. Verified element-wise
against an independent JAX Hartree-Fock implementation (slaterform) to
machine precision on individual integrals and to 10 significant figures
on Si2/STO-3G's full SCF energy.

`dashboard_core.hamiltonians` calls this engine automatically —
`bridge.build_qubit_hamiltonian` — whenever a requested molecule uses an
element outside PennyLane's own STO-3G table; existing molecules
(H2/HeH+/H3+/LiH/H2O) are unaffected and keep using PennyLane's `dhf`
pipeline directly. See [Dashboard Core — Hamiltonians](dashboard_core_hamiltonians.md)
for the dispatch logic and the Si2 catalog entry this engine backs.

::: dense_evolution.native_hf.bridge

::: dense_evolution.native_hf.scf

::: dense_evolution.native_hf.basis

---

**See also**: [`Dashboard Core — Hamiltonians`](dashboard_core_hamiltonians.md)
for the production entry point (`MOLECULE_CATALOG`'s Si2 entry), and
[`dashboard_core.vqe`](dashboard_core_vqe.md) for the ansatz circuits
optimized against Hamiltonians this engine can build.
