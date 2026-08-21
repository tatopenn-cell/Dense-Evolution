# Observables (Pauli-string expectation values)

Exact expectation values of Pauli strings, computed directly from a statevector in O(dim) via
bit manipulation -- the 2^n_qubits Hamiltonian matrix is never built. `pauli_sum_matvec` uses
the same technique for `H @ vector` (not reduced to a scalar) -- the matrix-free primitive
behind [`ground_state_energy_sparse`](dashboard_core_hamiltonians.md)'s
`scipy.sparse.linalg.eigsh` path for molecules too large to diagonalize densely.

::: dense_evolution.physics.observables

---

**See also**: [`circuit_to_energy_fn`](autodiff.md) for the differentiable, `h_matrix @ statevector`
approach to expectation values (needed for `jax.grad`-based VQE, at the cost of an explicit
dense Hamiltonian matrix) -- use `pauli_expectation`/`pauli_sum_expectation` instead whenever a
dense Hamiltonian isn't otherwise needed, or the system is too large for one to be practical.
