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
(4, True, 3, -1.116684335260025)
```

`build_qubit_hamiltonian(atomic_numbers, geometry_angstrom, n_electrons)` runs the full
pipeline -- Obara-Saika integrals, then Hartree-Fock self-consistent-field iteration
(`run_scf`, DIIS-accelerated) -- for H2 (two protons, atomic number 1, `0.7414` Angstrom
apart, the real equilibrium bond length) in the default STO-3G minimal basis. It
converged in 3 iterations to a mean-field energy of `-1.1167` Ha. `H` is a `qml.Hamiltonian` -- the
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
-1.1372701878105904
```

Converting `H`'s Pauli terms to [`pauli_hamiltonian_to_matrix`](observables.md)'s format
and exactly diagonalizing gives `-1.1373` Ha -- lower than Step 1's Hartree-Fock energy
by about `0.02` Ha, the real electron-correlation energy Hartree-Fock's single-Slater-
determinant approximation misses entirely. At only 4 qubits, exact diagonalization is
easy; that gap is exactly what a real VQE ansatz beyond a bare Hartree-Fock reference
state is trying to close for larger, classically-intractable molecules.

## Step 3. Large mixed-basis molecules: the optional libcint bridge

```python
import numpy as np
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.assembly import build_overlap_matrix, build_core_hamiltonian
from dense_evolution.native_hf.libcint_bridge import build_repulsion_tensor_libcint
from dense_evolution.native_hf.scf import run_scf

geometry = np.array([[0.0, 0.0, 0.0]])
shells = build_molecule_shells([10], geometry, "6-31g*")
S = build_overlap_matrix(shells)
H_core = build_core_hamiltonian(shells, [10.0], geometry)
V = build_repulsion_tensor_libcint([10], geometry, "6-31g*")

result = run_scf(S, H_core, V, 10, [10.0], geometry)
result.converged, result.total_energy
```

```
(True, -128.4744065199...)
```

Step 1's molecule only needed s functions, so `build_repulsion_tensor` (the
default, pure-JAX path) was already fast. A basis mixing s, p, and d shells
-- like `6-31g*` on neon here -- costs much more with the default path: the
underlying JIT compiles one program per distinct shell-quartet shape it
meets, and a mixed basis needs dozens of them. `build_repulsion_tensor_libcint`
computes the identical tensor through [PySCF](https://pyscf.org/)'s own
`libcint` instead -- a mature C library with no per-basis compile cost at
all -- and is a drop-in replacement everywhere `build_repulsion_tensor`'s
output was used (`S` and `H_core` above still come from native_hf itself).
Requires the `libcint` extra (`pip install dense-evolution[libcint]`); no
Windows wheel exists for PySCF, so that extra needs a C toolchain there.

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
individual integrals. The original Si2/STO-3G full-SCF-energy cross-check against
slaterform predates the DIIS/Si2-convergence fix below and has not been re-run against
the corrected energy -- the per-integral cross-check is unaffected (integrals don't
depend on how the SCF loop converges), but the end-to-end Si2 number should be treated
as not yet re-verified against this independent reference. Algorithm background also
drawn from PennyLane's own white paper (Delgado et al., "Differentiable quantum
computational chemistry with PennyLane", [arXiv:2111.09967](https://arxiv.org/abs/2111.09967)).

**The libcint bridge's AO-convention mismatch**: native_hf and PySCF/libcint agree on
the physics but not on bookkeeping -- confirmed empirically (`mol.ao_labels()`,
`mol.intor('int1e_ovlp')`), not assumed from either codebase's docs. Three separate
mismatches, all handled by `libcint_bridge.py`: (1) shell order -- native_hf keeps the
basis file's own order (s,p,s,p,d, interleaved on Ne/6-31G*), libcint groups all shells
of an atom by ascending degree (s,s,s,p,p,d) -- matched by identity (atom, degree,
exponents), not by assuming either list's order; (2) Cartesian component order --
native_hf's own `cartesian_powers` uses px,pz,py and xx,xz,xy,zz,yz,yy, libcint uses the
more common px,py,pz and xx,xy,xz,yy,yz,zz; (3) normalization -- libcint's raw d
components are not unit-self-overlap the way native_hf's are (xx/yy/zz vs. xy/xz/yz
differ by exactly a factor of 3 on Ne/6-31G*), corrected via a rescale computed from
libcint's own overlap diagonal at call time, not a hardcoded constant.

**SCF convergence (`run_scf`)**: DIIS-accelerated (Pulay 1980/1982) by default, not
plain linear damping -- see [the module's own docstring](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/dense_evolution/native_hf/scf.py)
for the real Si2 near-degenerate-orbital case that motivated this (11 DIIS iterations
vs. 53 for damping alone, same converged energy to 12 significant figures) and requires
both density and energy to stop changing before declaring convergence, not density
alone.

**AO-to-MO integral transformation**: `bridge._ao_to_mo`'s 4-index `einsum` +
`swapaxes` (converting AO-basis integrals to the molecular-orbital basis, using the
converged Hartree-Fock coefficients) is exactly the kind of operation where an
index-ordering mistake can produce a plausible-looking but numerically wrong
Hamiltonian -- cross-checked against two independent references: a sequential
one-index-at-a-time transform (a different algorithm computing the same quantity, not
a copy of the code under test) to `1e-10`, and a real physical invariant on H2 (a
basis change alone cannot alter the total electronic energy) to `1e-10`.

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

::: dense_evolution.native_hf.libcint_bridge
