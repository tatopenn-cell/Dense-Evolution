# Building Hamiltonians

## A little history first

In 1788, Joseph-Louis Lagrange rewrote Newton's mechanics around a single
function, the Lagrangian `L = T - V` (kinetic energy minus potential
energy), from which every equation of motion for a system falls out by a
single recipe. Fifty years later, William Rowan Hamilton reorganized the
same physics around a different function: the Hamiltonian `H = T + V`,
the system's *total* energy, expressed not in terms of position and
velocity but position and momentum. Hamilton's version turned out to be
the one that survived into the 20th century: when quantum mechanics was
built, it was Hamilton's `H` — not Lagrange's `L` — that became an
*operator*, and the Schrödinger equation `H|psi> = E|psi>` is exactly
Hamilton's old energy function, now acting on a quantum state instead of
a point in phase space.

So a Hamiltonian, in this library, is always the same idea: a Hermitian
matrix whose eigenvalues are the possible energies of a system, and
whose eigenvectors are the states that have those energies. Building one
from scratch means writing down that matrix — usually not by typing out
every entry, but by describing it as a sum of simple pieces.

## Step 1. The simplest possible Hamiltonian

```python
import numpy as np
import dense_evolution as de

terms = [(1.0, {0: 'Z'})]
H = de.pauli_hamiltonian_to_matrix(terms, 1)
print(H.real)
print(np.linalg.eigvalsh(H))
```

```
[[ 1.  0.]
 [ 0. -1.]]
[-1.  1.]
```

`pauli_hamiltonian_to_matrix` takes a list of `(coefficient, pauli_dict)`
terms — here just one term, coefficient `1.0`, acting as `Z` on qubit 0 —
and builds the full matrix. A single `Z` is already a genuine Hamiltonian:
its eigenvalues (`-1`, `+1`) are the two energies of a qubit that prefers
to point up or down, exactly the textbook two-level system every
quantum-mechanics course starts with.

## Step 2. A sum of terms

```python
terms = [(1.0, {0: 'X', 1: 'X'}), (1.0, {0: 'Y', 1: 'Y'}), (1.0, {0: 'Z', 1: 'Z'})]
H = de.pauli_hamiltonian_to_matrix(terms, 2)
print(np.linalg.eigvalsh(H))
```

```
[-3.  1.  1.  1.]
```

This is the two-qubit Heisenberg exchange Hamiltonian `H = X0X1 + Y0Y1 +
Z0Z1` — three terms instead of one, added together the same way any real
Hamiltonian is: term by term. Its ground energy `-3` corresponds to the
singlet state, three units below the triplet's `+1` — the real reason two
coupled spins prefer to anti-align, derived here from four numbers, not
asserted.

Every Hamiltonian in this library, however large, is built the same way:
decide which Pauli terms belong, list their coefficients, and let
`pauli_hamiltonian_to_matrix` (or, for a system too large to hold as a
dense matrix, [`pauli_sum_matvec`](api/observables.md) for the
matrix-free `H @ vector` version) do the assembly.

## Step 3. Fermionic systems: Jordan-Wigner

Not every system is naturally described by qubits — the Heisenberg
Hamiltonian above happens to already be a spin system. Electrons, and the
Majorana fermions used in models like SYK (Step 5), are a different kind
of object, and need one extra step before they become a Pauli sum:
Jordan-Wigner mapping.

```python
from dense_evolution.fermions import majorana_pauli_terms

coeff1, term1 = majorana_pauli_terms(1, 2)
coeff2, term2 = majorana_pauli_terms(2, 2)
print(coeff1, term1)
print(coeff2, term2)
```

```
1.0 {0: 'X'}
1.0 {0: 'Y'}
```

`majorana_pauli_terms(mode_index, n_qubits)` gives the Pauli representation
of one Majorana fermion mode — `chi_1 = X_0`, `chi_2 = Y_0` here, with a
`Z`-string prepended for higher-indexed modes so that different modes
anticommute correctly (`{chi_a, chi_b} = 2*delta_ab*I`, verified against
the real matrices in [Fermions](api/fermions.md)). Multiply several of
these together (see `dashboard_core.wormhole._multiply_pauli_dicts` for a
real, reusable Pauli-string-multiplication helper) and feed the result to
`pauli_hamiltonian_to_matrix`, and you can build any Majorana-operator
Hamiltonian the same way Step 2 built a spin Hamiltonian.

## Step 4. Real molecules: native Hartree-Fock

For an actual molecule, the Hamiltonian's terms aren't something you'd
want to write by hand — they come from solving the electronic structure
problem first. `dense_evolution.native_hf` does this from scratch (real
Gaussian-orbital integrals via the Obara-Saika recursion, then a Roothaan-Hall
self-consistent-field loop), and hands the converged result to
`build_qubit_hamiltonian`:

```python
from dense_evolution.native_hf.bridge import build_qubit_hamiltonian

hamiltonian, n_qubits, hf_result = build_qubit_hamiltonian(
    atomic_numbers=[1, 1],
    geometry_angstrom=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
    n_electrons=2,
    basis_name="sto-3g",
)
print(n_qubits, round(float(hf_result.total_energy), 6), hf_result.converged)
```

```
4 -1.116759 True
```

Four qubits, a converged SCF energy of `-1.116759` Hartree for H2 at its
equilibrium bond length (0.74 Angstrom) — the same real molecule the
[VQE example](getting-started.md#quick-start) later optimizes over. See
[Native Hartree-Fock](api/native_hf.md) for the full parameter reference
(active-space selection, basis sets, the `HFResult` fields) and
[Dashboard Core — Hamiltonians](api/dashboard_core_hamiltonians.md) for
the molecule catalog this engine backs.

## Step 5. Something genuinely custom: an SYK Hamiltonian

Steps 1-4 all used Hamiltonians with a name (Heisenberg, a molecule's
electronic Hamiltonian). Nothing stops you from inventing your own,
combining the exact same two primitives — `majorana_pauli_terms` and
`pauli_hamiltonian_to_matrix`. Here is a real four-body Sachdev-Ye-Kitaev
(SYK) Hamiltonian, `H = sum_{i<j<k<l} J_ijkl * chi_i*chi_j*chi_k*chi_l`,
built from every 4-Majorana combination on 8 modes with random Gaussian
couplings:

```python
import itertools
import sys
sys.path.insert(0, "tools/dashboard")
from core.wormhole import _multiply_pauli_dicts

def build_syk4_terms(n_majorana, J, seed):
    n_qubits = n_majorana // 2
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(6.0 * J**2 / n_majorana**3)
    terms = []
    for quad in itertools.combinations(range(1, n_majorana + 1), 4):
        dicts = [majorana_pauli_terms(m, n_qubits)[1] for m in quad]
        phase, merged = _multiply_pauli_dicts(dicts)
        terms.append((rng.normal(0.0, sigma) * phase, merged))
    return n_qubits, terms

n_qubits, terms = build_syk4_terms(n_majorana=8, J=5.0, seed=61)
H = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
print(n_qubits, len(terms), np.allclose(H, H.conj().T))
```

```
4 70 True
```

Four qubits (`8` Majorana modes, two per qubit), `70` terms (every
4-combination of 8 modes, `C(8,4)=70`), and a genuinely random-looking
Hamiltonian that is nonetheless exactly Hermitian by construction — the
same pattern the [Chunk](api/chunk.md) and wormhole-teleportation
experiments build on internally, just assembled here from first
principles instead of hidden behind a ready-made function.

---

## Details

### Coupling two independent Hamiltonians: the Klein-factor trap

A natural next step past Step 5 is coupling *two* Hamiltonians together —
e.g. a "left" and "right" copy of the same SYK model, as in
traversable-wormhole teleportation protocols. This looks like it should
be as simple as building each side's Majoranas independently and shifting
one side's qubit indices, but it silently breaks: a fermionic number
operator like `n = c^dagger c`, built from one Majorana on each side
(`c = (chi_L + i*chi_R)/2`), needs `chi_L` and `chi_R` to *anticommute*.
Two independently Jordan-Wigner-mapped registers, tensored together,
automatically *commute* instead (they act on disjoint qubits) — and under
that convention `n` collapses to the trivial constant `0.5 * I`, not a
real 0-or-1-valued operator (verified directly: the resulting coupling
unitary fails to be unitary, `||U^dagger U - I|| ~ 55`, regardless of
which sign convention is tried for the cross term). This is the standard
"Klein factor" problem in the SYK/wormhole literature — coupling two
independently-built fermionic registers needs an explicit parity-string
correction, not a sign tweak. If you're building a two-copy coupled
Hamiltonian, look up how the source paper you're reproducing handles this
explicitly; don't assume independent registers can simply be tensored
together.

### When the matrix is too large to build

Every example above builds a dense `2**n_qubits` matrix. Past roughly
20-25 qubits this stops being practical — use
[`pauli_sum_matvec`](api/observables.md) (matrix-free `H @ vector`,
the same technique behind `ground_state_energy_sparse`'s
`scipy.sparse.linalg.eigsh` path for large molecules) or
[`circuit_to_energy_fn`](api/autodiff.md) if you need the energy as a
differentiable function of circuit parameters rather than the bare
matrix.

## See Also

- [Observables](api/observables.md) — `pauli_hamiltonian_to_matrix`,
  `pauli_sum_matvec`.
- [Fermions](api/fermions.md) — `majorana_pauli_terms`, the Jordan-Wigner
  mapping used in Steps 3 and 5.
- [Native Hartree-Fock](api/native_hf.md) — the full molecular-Hamiltonian
  pipeline Step 4 only introduces.
- [Autodiff](api/autodiff.md) — `circuit_to_energy_fn`, for using a
  Hamiltonian built here inside a differentiable VQE loop.
