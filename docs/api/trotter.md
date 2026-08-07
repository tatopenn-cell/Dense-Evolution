# Trotter (real-time Hamiltonian evolution as gates)

Real-time Hamiltonian evolution as an actual gate circuit
(Trotterization) — did not exist anywhere in this package before. Every
existing piece of "evolution" machinery elsewhere is either
gate-based-and-fixed (a hand-written or VQE-optimized circuit template)
or exact-and-not-a-circuit (`dashboard_core.hamiltonians.ground_state_energy`'s
dense diagonalization). Nothing composed `exp(-i*H*t)` for an arbitrary
Hamiltonian into gates a real quantum computer could run.

`pauli_rotation_ops` is exact for a single Pauli-string term (fidelity
1.0 against `scipy.linalg.expm`, verified for 1-4 qubit mixed X/Y/Z
strings, not just Z-strings); `trotter_evolve_ops` composes many such
terms via the first-order product formula, an *approximation* whose
error shrinks as `n_steps` grows — verified: infidelity drops roughly 4x
per doubling of steps against a real, non-trivial multi-qubit
Hamiltonian, consistent with the expected quadratic convergence of
first-order Trotter error in state overlap.

::: dense_evolution.trotter

---

**See also**: [`fermions`](fermions.md) and [`entropy`](entropy.md), the
other two modules promoted alongside this one from a real traversable-
wormhole-inspired quantum teleportation reproduction (arXiv:2604.10090).
`dashboard_core.wormhole.run_wormhole_protocol_trotter` uses this
module's Trotterized circuit as the "closer to real hardware" backend,
cross-verified against the exact-evolution backend — see
[Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the real experiments (run with the exact backend, for scan speed).
