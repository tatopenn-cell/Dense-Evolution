# Fermions (Majorana Jordan-Wigner mapping)

Majorana-fermion → qubit (Jordan-Wigner) mapping, one qubit per two
Majorana modes: `chi_{2j-1} = (prod_{k<j} Z_k) X_j`,
`chi_{2j} = (prod_{k<j} Z_k) Y_j`. Each `chi_i` is Hermitian and unit-
normalized (`chi_i^2 = I`), and the anticommutation relation
`{chi_a, chi_b} = 2*delta_ab*I` holds exactly — verified against the
actual matrices, not assumed from the textbook formula alone.

Combine the returned Pauli term with
[`pauli_hamiltonian_to_matrix`](observables.md) to build any
Majorana-operator Hamiltonian as a dense matrix, e.g. a sparse
Sachdev-Ye-Kitaev (SYK) model: `H = sum_{ijkl} J_ijkl * chi_i*chi_j*chi_k*chi_l`.

`total_parity_operator` builds the "Klein factor" for a set of Majorana
modes — the tool needed when TWO independently-Jordan-Wigner-mapped
registers (e.g. the two sides of a thermofield-double/wormhole
construction) must be combined into one joint fermionic algebra: their
Majoranas commute across registers by construction (disjoint qubits),
but a genuine cross-register Dirac fermion needs them to anticommute.
Dressing one register's operators with its own `total_parity_operator`
before combining fixes this — see the function's own docstring for the
full derivation, and
[Dense-Evolution-Discovery's wormhole_magic_entropy.py](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/)
for the real construction this was promoted from.

`hubbard_hamiltonian_pauli_terms` uses the OTHER standard Jordan-Wigner
convention — ordinary spin-orbital creation/annihilation operators
(`c_q = sigma+_q * Z-string`, not Majoranas) — to map the 1D Hubbard-ring
Hamiltonian `H = -t*sum_<ij>,sigma (c^dagger_i c_j + h.c.) + U*sum_i
n_i,up*n_i,down` onto Pauli terms. With `n_sites=4` and `periodic=True`
this is the "Hubbard square" studied in Arovas, Bandyopadhyay & Zhu,
"The Hubbard Model" (Annual Review of Condensed Matter Physics 2022,
arXiv:2103.12097) — Table 2 (p.6) gives a closed-form small-`U/t`
perturbative ground-state energy for this exact model, and identifies
its ground state's orbital symmetry as `x^2-y^2` (B1g/d-wave). See
[Dense-Evolution-Discovery's hubbard_square_arovas.py](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/)
(Experiment 38) for the full verification: the perturbative formula
checked directly against the paper's own text, and the periodic
wraparound bond (the one place a naive Jordan-Wigner implementation
could plausibly need an extra parity correction) checked against an
independent brute-force fermionic construction — machine-exact
agreement, not assumed from the formula alone.

::: dense_evolution.physics.fermions

---

**See also**: [`entropy`](entropy.md) and [`trotter`](trotter.md), the
other two modules promoted alongside this one from a real traversable-
wormhole-inspired quantum teleportation reproduction (Gao-Jafferis-Wall
theory, arXiv:2604.10090) — see the [MCP Server](../composer.md) section
and [Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the real experiments built on top of these three modules.
