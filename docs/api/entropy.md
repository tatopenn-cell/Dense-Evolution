# Entropy (partial trace, von Neumann entropy, mutual information)

A qubit entangled with another one has no well-defined state of its own -- only the
*pair* has a pure state. Measuring "how mixed" that single qubit looks on its own is
exactly how entanglement shows up numerically: a qubit maximally entangled with its
partner looks maximally random in isolation, even though the two-qubit system as a
whole is perfectly pure. This module measures that: reducing a multi-qubit state down
to a subsystem (`partial_trace`), quantifying how mixed the result is
(`von_neumann_entropy`), and how much two subsystems know about each other
(`mutual_information`).

## Step 1. Reduce a Bell state down to one qubit

```python
import numpy as np
import dense_evolution as de
from dense_evolution.physics.entropy import partial_trace

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = sim.get_statevector()

rho0 = partial_trace(sv, n_qubits=2, keep_qubits=[0])
np.round(rho0, 4)
```

```
array([[0.5+0.j, 0. +0.j],
       [0. +0.j, 0.5+0.j]])
```

`sv` is the same Bell state built on the [Simulator](simulator.md) page. `partial_trace`
"forgets" qubit 1 and returns qubit 0's own `2x2` density matrix -- `I/2`, maximally
mixed. That's the signature of entanglement: qubit 0 alone looks like a fair coin flip,
even though the full 2-qubit state is perfectly pure and deterministic.

## Step 2. How mixed is that, exactly?

```python
from dense_evolution.physics.entropy import von_neumann_entropy

von_neumann_entropy(rho0)
```

```
0.6931471805599454
```

`von_neumann_entropy` is `-Tr(rho * log(rho))`, the quantum generalization of Shannon
entropy -- `0` for a pure state, and `ln(2) = 0.6931...` (matching the printed value)
for a single qubit's *maximum* possible mixedness. Step 1's `rho0` hit that ceiling
exactly, which is only possible when qubit 0 is maximally entangled with the rest of
the system.

## Step 3. Do the two qubits know about each other?

```python
from dense_evolution.physics.entropy import mutual_information

mutual_information(sv, n_qubits=2, qubits_a=[0], qubits_b=[1])
```

```
1.3862943611189236
```

`mutual_information(state, n_qubits, qubits_a, qubits_b)` measures correlation between
two subsystems directly from the full state -- no need to call `partial_trace` on each
side yourself first. `2*ln(2) = 1.3862...` is the maximum two qubits can share, and a
Bell pair hits it exactly. This is the tool for a case a single-qubit expectation value
structurally cannot catch: a qubit entangled in a Bell pair (or any subsystem
maximally mixed on its own) has `<Z> = 0` regardless of what happened to its partner --
the no-signaling theorem, not a measurement limitation -- but its mutual information
with that partner is very much nonzero.

## Step 4. Fitting a whole entropy curve to CFT theory

```python
from dense_evolution.physics.entropy import central_charge

N = 20
Ls = [2, 4, 6, 8, 10]
S = [(0.5 / 6.0) * np.log((2 * N / np.pi) * np.sin(np.pi * L / N)) + 0.3 for L in Ls]

central_charge(Ls, S, n_qubits=N)
```

```
(0.5000000000000007, 1.0)
```

`central_charge(Ls, S, n_qubits)` fits an already-measured entanglement-entropy curve
`S(L)` (entropy of the first `L` qubits, at several subsystem sizes `L`) to the
Calabrese-Cardy CFT prediction and returns `(c, r_squared)` -- backend-agnostic, it
doesn't compute entropies itself, only fits a curve you already measured (e.g. via
Steps 1-2 above, repeated at each `L`). `S` here is built directly from the
Calabrese-Cardy formula at the known value `c=0.5`, so a perfect round-trip
(`r_squared=1.0`, recovered `c` matching to `1e-15`) confirms the fit itself is
correct -- a real critical spin chain's measured entropies won't be this clean, but the
fit machinery is the same either way.

---

## Details

**Indexing convention**: qubit 0 is the *most* significant bit of the basis-state index
throughout this module, matching [`observables`](observables.md)/
`pauli_hamiltonian_to_matrix` -- not the little-endian convention some other libraries
use. The only prior partial trace in this package before this module existed
(`dashboard_core/state_visuals.py`'s private, single-qubit-only `_reduced_density_matrix`)
used the *opposite* convention -- do not reuse that helper here, it would silently
transpose which qubits get traced out.

**A high `r_squared` alone doesn't mean the extracted `c` is trustworthy**:
Dense-Evolution-Discovery [Experiment 36](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/central_charge_calabrese_cardy/)
found that fitting a real critical Ising chain's entropy curve at a finite-size
*susceptibility-peak* pseudo-critical point (instead of the true self-dual CFT point)
gives a deceptively clean fit (`r_squared=0.999997`) to a wrong answer -- `c` off by
roughly 2x from the known Ising value `c=1/2`. Fitting at the correct critical point
recovered `c=0.565`, much closer to the true `0.5`.

::: dense_evolution.physics.entropy

---

**See also**: [`fermions`](fermions.md) and [`trotter`](trotter.md), the other two
modules promoted alongside this one from a real traversable-wormhole-inspired quantum
teleportation reproduction (arXiv:2604.10090) -- see
[Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the real experiments, including a control run confirming `mutual_information`
correctly returns exactly `0` when two subsystems are structurally disconnected.
