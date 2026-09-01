# Random Circuit (benchmarking & fuzz-testing)

Sometimes the circuit itself doesn't matter -- only that it's a real, valid one, for
timing a backend or fuzz-testing a parser against whatever gate combinations show up.
`random_circuit` fills that role, the same one Qiskit's own `random_circuit` plays.

## Step 1. A reproducible random circuit

```python
import dense_evolution as de

ops = de.random_circuit(n_qubits=3, n_gates=6, seed=0)
ops
```

```
[('t', 0), ('cx', 0, 2), ('t', 1), ('tdg', 1), ('z', 2), ('cz', 2, 1)]
```

`random_circuit(n_qubits, n_gates, seed=None)` returns a plain gate-tuple list, the
same shape `run_circuit_jit` expects everywhere else in this package -- `seed=0` makes
the draw reproducible (the exact 6 gates above, every time); omit it for a fresh random
circuit each call. `two_qubit_prob` (default `0.4`) controls roughly what fraction of
gates are two-qubit rather than single-qubit; `gate_set` restricts which named gates
are eligible to be drawn, if only a subset should show up.

## Step 2. It's a real circuit -- run it

```python
sim = de.DenseSVSimulator(3)
sim.run_circuit_jit(ops)
sim.get_probabilities().round(4)
```

```
array([1., 0., 0., 0., 0., 0., 0., 0.])
```

Every gate `random_circuit` draws is real and runs exactly like a hand-written one --
here the whole probability mass lands back on `|000>` (this particular random draw
happens to be entirely `Z`-basis-diagonal gates plus phase gates starting from `|000>`,
so no amplitude ever moves off the all-zero state). A different `seed`, or more/
different gates, would spread probability across other basis states instead -- the
draw is real, not guaranteed to look "interesting" every time.

---

## Details

**What it's for**: benchmarking (a real circuit shape to time a backend against,
without hand-writing one for every qubit count/depth combination) and fuzz-testing (a
parser or compiler seeing gate combinations a human wouldn't necessarily think to write
by hand).

::: dense_evolution.random_circuit
