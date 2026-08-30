# Entropy (partial trace, von Neumann entropy, mutual information)

Multi-qubit partial trace, von Neumann entropy, and quantum mutual
information — nothing like this existed anywhere in the package before
these functions were promoted. The only prior partial trace
(`dashboard_core/state_visuals.py`'s private `_reduced_density_matrix`)
is single-qubit-only and uses the *opposite*, little-endian convention
(qubit 0 = least significant bit). This module uses the package's own
convention instead, matching [`observables`](observables.md)/
`pauli_hamiltonian_to_matrix`: qubit 0 is the *most* significant bit of
the basis-state index — do not mix the two, reusing the dashboard's
helper here would silently transpose which qubits get traced out.

`mutual_information` exists because a qubit entangled in a Bell pair (or
more generally, maximally mixed on its own) has a marginal `<Z>` of
exactly 0 regardless of what operation was applied to its partner — the
no-signaling theorem, not a measurement limitation. Mutual information
*can* reveal correlations a marginal expectation value structurally
cannot, since it depends on the joint state of two subsystems, not
either one alone. Verified against the exact textbook value for a Bell
pair (`I = 2*ln(2)`, maximal) and a GHZ state.

`central_charge` fits an open-chain entanglement entropy curve `S(L)` to
the Calabrese-Cardy CFT prediction (Calabrese & Cardy, J. Stat. Mech.
2004, P06002) and returns the extracted central charge plus fit quality
(`r_squared`) -- backend-agnostic, meant as a benchmark diagnostic for
whether a given simulator backend or truncation scheme preserves genuine
critical CFT scaling. Promoted from
[Dense-Evolution-Discovery Experiment 36](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/central_charge_calabrese_cardy/),
which recovered the known Ising CFT value `c=1/2` from a real critical
transverse-field Ising chain and, along the way, found and documented a
real pitfall: fitting at a finite-size *susceptibility-peak* pseudo-
critical point instead of the true self-dual CFT point gives a
deceptively clean fit (`r_squared=0.999997`) to a wrong answer (`c`
off by 2x) -- a high `r_squared` alone does not mean the extracted `c` is
trustworthy.

::: dense_evolution.physics.entropy

---

**See also**: [`fermions`](fermions.md) and [`trotter`](trotter.md), the
other two modules promoted alongside this one from a real traversable-
wormhole-inspired quantum teleportation reproduction (arXiv:2604.10090)
— see [Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the real experiments, including a control run confirming
`mutual_information` correctly returns exactly `0` when two subsystems
are structurally disconnected.
