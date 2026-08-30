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

::: dense_evolution.physics.fermions

---

**See also**: [`entropy`](entropy.md) and [`trotter`](trotter.md), the
other two modules promoted alongside this one from a real traversable-
wormhole-inspired quantum teleportation reproduction (Gao-Jafferis-Wall
theory, arXiv:2604.10090) — see the [MCP Server](../composer.md) section
and [Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the real experiments built on top of these three modules.
