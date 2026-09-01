# Native Hartree-Fock

Before a molecule's electronic-structure problem can become a qubit Hamiltonian, its
integrals (overlap, kinetic, nuclear-attraction, electron-repulsion) have to be
computed and a Hartree-Fock mean-field calculation run to get a reference orbital
basis. `dense_evolution.native_hf` is a from-scratch, JAX-vectorized engine for that
step -- covers any element `basis_set_exchange` has STO-3G data for, not just the small
H-Ne table PennyLane's own bundled solver ships.

## Step 1. A real molecule's qubit Hamiltonian

```python
import numpy as np
from dense_evolution.native_hf.bridge import build_qubit_hamiltonian

geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]])
H, n_qubits, hf_result = build_qubit_hamiltonian([1, 1], geometry, n_electrons=2)

n_qubits, hf_result.converged, hf_result.n_iterations, hf_result.total_energy
```

```
(4, True, 13, -1.1166843352600253)
```

`build_qubit_hamiltonian(atomic_numbers, geometry_angstrom, n_electrons)` runs the full
pipeline -- Obara-Saika integrals, then Hartree-Fock self-consistent-field iteration
(`run_scf`) -- for H2 (two protons, atomic number 1, `0.7414` Angstrom apart, the real
equilibrium bond length) in the default STO-3G minimal basis. It converged in 13
iterations to a mean-field energy of `-1.1167` Ha. `H` is a `qml.Hamiltonian` -- the
converged result still goes through PennyLane's own `fermionic_observable` +
`jordan_wigner` for the qubit mapping, since that stage was already fast and
well-tested; this module only replaces the slow integral/SCF stage.

## Step 2. Hartree-Fock is a mean-field approximation -- how far off?

```python
import dense_evolution as de

coeffs, ops = H.terms()
terms = []
for coeff, op in zip(coeffs, ops):
    pauli = {}
    for factor in (op.operands if hasattr(op, 'operands') else [op]):
        if factor.name != 'Identity':
            pauli[int(factor.wires[0])] = factor.name[-1]
    terms.append((float(np.real(complex(coeff))), pauli))

H_dense = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
float(np.min(np.linalg.eigvalsh(H_dense)))
```

```
-1.1372701878105915
```

Converting `H`'s Pauli terms to [`pauli_hamiltonian_to_matrix`](observables.md)'s format
and exactly diagonalizing gives `-1.1373` Ha -- lower than Step 1's Hartree-Fock energy
by about `0.02` Ha, the real electron-correlation energy Hartree-Fock's single-Slater-
determinant approximation misses entirely. At only 4 qubits, exact diagonalization is
easy; that gap is exactly what a real VQE ansatz beyond a bare Hartree-Fock reference
state is trying to close for larger, classically-intractable molecules.

---

## Details

**Why this module exists**: PennyLane's own differentiable Hartree-Fock solver
(`qml.qchem`, `method="dhf"`) builds the same integrals through a Python-level loop
wrapped in its autograd-tracing numpy layer -- correct, but profiled directly at 482 of
483 total seconds for Si2/STO-3G, almost entirely per-scalar-op tracer overhead rather
than real FLOPs. This module batches each shell-pair/quartet with `jax.lax.scan`/
`jax.vmap` and compiles with `jax.jit` instead.

**Elements beyond PennyLane's table**: basis-set parameters come from
[`basis_set_exchange`](https://github.com/MolSSI-BSE/basis_set_exchange), so any
element it has STO-3G data for is reachable -- an element needing d-orbitals or higher
(e.g. Fe) fails with a clear `NotImplementedError` naming the real limitation, not a
silent wrong energy for an incomplete basis.

**Verified against an independent implementation**: element-wise against
[lowdanie/hartree-fock-solver](https://github.com/lowdanie/hartree-fock-solver)
("slaterform", Apache-2.0, studied as a reference for structuring the Obara-Saika
recursion with `jax.lax.scan` -- no source code copied) to machine precision on
individual integrals, and to 10 significant figures on Si2/STO-3G's full SCF energy.
Algorithm background also drawn from PennyLane's own white paper (Delgado et al.,
"Differentiable quantum computational chemistry with PennyLane",
[arXiv:2111.09967](https://arxiv.org/abs/2111.09967)).

**Production entry point**: [`dashboard_core.hamiltonians`](dashboard_core_hamiltonians.md)
calls this engine automatically (`bridge.build_qubit_hamiltonian`) whenever a requested
molecule uses an element outside PennyLane's own STO-3G table -- existing catalog
molecules (H2/HeH+/H3+/LiH/H2O) are unaffected and keep using PennyLane's `dhf`
pipeline directly. See that page for the dispatch logic and the Si2 catalog entry this
engine backs, and [`dashboard_core.vqe`](dashboard_core_vqe.md) for ansatz circuits
optimized against Hamiltonians built this way.

::: dense_evolution.native_hf.bridge

::: dense_evolution.native_hf.scf

::: dense_evolution.native_hf.basis
