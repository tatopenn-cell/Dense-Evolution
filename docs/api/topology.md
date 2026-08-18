# Topology (entangling-layer patterns)

Named entangling-layer connectivity patterns for variational circuits (`linear`, `circular`,
`full`, `star`, `brick`) -- the same role Qiskit's `TwoLocal(entanglement=...)` or PennyLane's
`qml.broadcast(pattern=...)` play, returning a plain gate-tuple list ready for `run_circuit`.

::: dense_evolution.circuits.topology

---

**See also**: [`ghz_state`](states.md) builds its CX chain with `entangling_layer(pattern='linear')`
directly. This package's `DenseSVSimulator` has no notion of hardware connectivity at all (any
two qubits can always interact) -- these patterns are an ansatz-design convenience, not a
constraint the simulator enforces.
