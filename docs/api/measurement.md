# Measurement (shot sampling & pure-state fidelity)

A real quantum device never hands back a statevector -- it hands back a stream of
individual measurement outcomes, one per shot, that you have to run many times and
count up to approximate the underlying distribution. `sample_counts` does that
simulated-device step (a `DenseSVSimulator` equivalent of Qiskit's `get_counts()`).
`statevector_fidelity` answers a different question: how close are two *known* pure
states to each other -- useful for grading a result against an ideal target, something
no real device measurement alone can tell you.

## Step 1. Finite-shot counts from a real statevector

```python
import numpy as np
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = sim.get_statevector()

de.sample_counts(sv, n_shots=1000, rng=np.random.default_rng(0))
```

```
{'00': 473, '11': 527}
```

`sv` is the same Bell state built on the [Simulator](simulator.md) page --
`sample_counts(statevector, n_shots, rng)` draws `n_shots` independent outcomes from
its exact Born-rule probabilities and returns a dict of bitstring counts, the same
shape `get_probabilities()` would predict exactly (`50/50` here) but with the real
sampling noise a finite number of shots actually has -- `473`/`527`, not `500`/`500`.
Passing `rng` makes the draw reproducible; omit it for a fresh random draw each call.

## Step 2. How close are two pure states?

```python
de.statevector_fidelity(sv, sv)
```

```
0.9999999999999996
```

```python
qasm2 = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; h q[1];'
sim2 = de.DenseSVSimulator(2)
sim2.run_circuit_jit(de.QASMParser().parse(qasm2).to_tuples())
sv2 = sim2.get_statevector()

de.statevector_fidelity(sv, sv2)
```

```
0.4999999999999998
```

`statevector_fidelity(a, b)` is `|<a|b>|^2`, computed directly on two statevectors --
no density matrix built first, unlike [`uhlmann_fidelity`](mitigation.md) (this
function's mixed-state counterpart). A state compared to itself gives `1` (up to
floating-point rounding); the Bell state against `|++>` (independent `H` on each qubit,
no entanglement) gives exactly `0.5` -- the two states share half their overlap, not
zero and not identical.

---

## Details

**When to use which fidelity**: `statevector_fidelity` needs both states to be pure
(exact statevectors, as any noiseless simulation produces) -- for a genuinely mixed
state (a real device's noisy output, or anything already reduced via
[`partial_trace`](entropy.md)), use [`uhlmann_fidelity`](mitigation.md) instead, which
takes density matrices.

::: dense_evolution.utils.measurement
