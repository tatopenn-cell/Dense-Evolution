# Fermions (Jordan-Wigner mapping)

A fermion (an electron, say) obeys the Pauli exclusion principle -- two of them can
never occupy the same state -- which shows up mathematically as *anticommutation*:
swapping two fermionic operators flips a sign, `a*b = -b*a`, unlike ordinary qubit
operators, which mostly commute. The Jordan-Wigner mapping is the standard recipe for
building qubit operators that reproduce this anticommuting behavior exactly, by
attaching a string of `Z` gates that tracks the "which fermions come before this one"
bookkeeping. This module builds two different flavors of that mapping: Majorana
operators (this page's main guide) and, for a specific Hubbard-model use case, ordinary
fermion creation/annihilation operators (Step 4).

## Step 1. A Majorana operator, and what makes it special

```python
import dense_evolution as de

de.majorana_pauli_terms(1, 2)
```

```
(1.0, {0: 'X'})
```

`majorana_pauli_terms(mode_index, n_qubits)` returns one Majorana fermion operator as
a `(coeff, {qubit: pauli})` term -- the same format
[`pauli_hamiltonian_to_matrix`](observables.md) and
[`pauli_sum_expectation`](observables.md) accept. `mode_index` runs from `1` to
`2*n_qubits` (two Majorana modes share every qubit); mode `1` on a 2-qubit register is
just `X` on qubit 0. Every Majorana operator is Hermitian and squares to the identity
(`chi_i^2 = I`) -- it behaves like a real, physical observable, not an abstract
placeholder.

## Step 2. Anticommutation, checked on a real state

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = sim.get_statevector()

from dense_evolution.physics.observables import pauli_sum_matvec

chi1 = de.majorana_pauli_terms(1, 2)
chi3 = de.majorana_pauli_terms(3, 2)

def apply(term, v):
    coeff, pauli = term
    return coeff * pauli_sum_matvec(v, [(1.0, pauli)], n_qubits=2)

anticommutator = apply(chi1, apply(chi3, sv)) + apply(chi3, apply(chi1, sv))
anticommutator
```

```
array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
```

`sv` is the same Bell state built on the [Simulator](simulator.md) page. `chi1` lives
entirely on qubit 0 (mode 1); `chi3` is mode 1 of the *second* Majorana pair, which
lands on qubit 1 but carries a leading `Z` on qubit 0 -- the Jordan-Wigner string. That
`Z` is exactly what makes `chi1` and `chi3` anticommute even though they act on
different qubits: applying both operators in either order to the Bell state gives the
exact zero vector, `{chi1, chi3}|psi> = 0` for every `|psi>`, the defining property of
two independent Majorana modes.

## Step 3. Combining two separate registers: the Klein factor

```python
de.total_parity_operator([1, 2, 3, 4], 2)
```

```
((1+0j), {0: 'Z', 1: 'Z'})
```

`total_parity_operator` multiplies every Majorana in a register together, which
collapses to the register's total-parity operator -- `Z` on every qubit in that
register (all 4 modes of a 2-qubit register, above, give `Z0 Z1`). It anticommutes with
every individual Majorana *in that same register*, the same way `chi1`/`chi3` did in
Step 2. That property is the tool needed when two *independently* Jordan-Wigner-mapped
registers (e.g. the two sides of a wormhole-teleportation construction) have to be
combined into one joint fermionic algebra: two Majoranas from different registers act
on disjoint qubits with no shared `Z`-string, so they naively *commute* -- wrong for a
genuine cross-register fermion. Dressing one register's operators with its own
`total_parity_operator` first restores the correct anticommutation across the join.

## Step 4. A different mapping, for the Hubbard model

```python
terms = de.hubbard_hamiltonian_pauli_terms(n_sites=2, t=1.0, U=2.0, periodic=False)
len(terms)
```

```
12
```

The Hubbard model (electrons hopping between sites, paying an energy penalty `U` for
two electrons sharing a site) needs ordinary creation/annihilation operators, not
Majoranas -- `hubbard_hamiltonian_pauli_terms` uses the *other* standard Jordan-Wigner
convention for that (`c_q = sigma+_q * Z-string`). `n_sites=2, periodic=False` is the
smallest non-trivial case: 2 lattice sites, 4 qubits (spin-up and spin-down per site).

```python
qasm4 = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; x q[0]; x q[2];'
circuit4 = de.QASMParser().parse(qasm4)
energy_fn, n_params = de.circuit_to_energy_fn(circuit4, n_qubits=4)

h_op = de.PauliSumOperator(terms, n_qubits=4)
theta = []
energy, sv = energy_fn(theta, h_op)
energy
```

```
2.0
```

`x q[0]; x q[2];` prepares both spin-up and spin-down electrons on site 0 -- one
doubly-occupied site, no electron anywhere else. No hopping is possible from a state
this localized (`t` never contributes), so the energy is exactly the interaction
penalty `U=2.0` for that one double occupancy -- [`PauliSumOperator`](observables.md)
applies `terms` directly to the statevector, the same differentiable path
[`circuit_to_energy_fn`](autodiff.md) uses for `jax.grad`-based VQE, without ever
building a dense Hamiltonian matrix.

---

## Details

**Indexing convention**: `mode_index` is 1-indexed (`1` to `2*n_qubits`), matching the
physics literature's `chi_1, chi_2, ...` convention -- `mode_index=0` raises
`ValueError`. Qubit assignment follows this package's usual most-significant-bit
convention throughout.

**Building a Hamiltonian from Majoranas**: combine several `majorana_pauli_terms`
results with [`multiply_pauli_terms`](observables.md) (e.g. a 4-Majorana product
`chi_i*chi_j*chi_k*chi_l` for a Sachdev-Ye-Kitaev-style term) and pass the resulting
terms list to [`pauli_hamiltonian_to_matrix`](observables.md) or
[`PauliSumOperator`](observables.md).

**Provenance**: `majorana_pauli_terms`/`total_parity_operator` were promoted from a
real traversable-wormhole-inspired quantum teleportation reproduction (Gao-Jafferis-Wall
theory, arXiv:2604.10090) -- see
[Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for that experiment. `hubbard_hamiltonian_pauli_terms` was promoted from Dense-Evolution-Discovery
Experiment 39 (the "Hubbard square", Arovas, Bandyopadhyay & Zhu, "The Hubbard Model",
Annual Review of Condensed Matter Physics 2022, arXiv:2103.12097), which checked the
paper's own Table 2 perturbative ground-state formula against exact diagonalization and
its predicted `x^2-y^2` (B1g/d-wave) ground-state symmetry against a real pairing-correlation
sign pattern -- see
[Dense-Evolution-Discovery's hubbard_square_arovas.py](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/hubbard_square_arovas/)
for the full writeup, including the independent brute-force check that the periodic
wraparound bond (the one place a naive Jordan-Wigner implementation could plausibly
need an extra parity correction) needs none.

::: dense_evolution.physics.fermions

---

**See also**: [`entropy`](entropy.md) and [`trotter`](trotter.md), the other two
modules promoted alongside `majorana_pauli_terms`/`total_parity_operator` from the same
wormhole-teleportation reproduction.
