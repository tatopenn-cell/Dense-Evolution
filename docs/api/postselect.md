# Postselection

Measuring a qubit gives a random outcome. Postselection keeps only the runs in which
the outcome is the one you chose, and throws the others away: the state becomes the
branch that matches, renormalised to length one. It is a tool for analysing circuits
and building heralded protocols; on real hardware it costs repeated runs, since the
discarded outcomes still happen.

## Step 1. Postselect one half of a Bell pair

```python
import numpy as np
import dense_evolution as de

qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""
circ = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circ)
sv0 = sim.get_statevector()
sv1, p = de.postselect(sv0, 2, 0, '+')
print(np.round(np.asarray(sv1), 4), round(p, 4))
```

```
[0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j] 0.5
```

`h` and `cx` build the Bell state `(|00> + |11>)/√2`. `postselect(sv0, 2, 0, '+')`
keeps the branch where qubit 0 is found in `|+>`. Because the two qubits are
entangled, qubit 1 is forced into `|+>` as well: all four amplitudes become `0.5`, the
state `|+>|+>`. The second number, `0.5`, is how often that outcome occurs, so half the
runs would be kept.

The target state can be `'0'`, `'1'`, `'+'`, `'-'`, `'+i'` or `'-i'`.

---

## Details

**Definition.** Postselection is the operation in the definition of PostBQP: the
computation is conditioned on a measurement outcome with nonzero probability
(Aaronson, *Quantum computing, postselection, and probabilistic polynomial-time*,
quant-ph/0412187, Definition 1). An outcome with zero probability raises
`ValueError`.

**How it is applied.** `postselect` multiplies the state by the projector
`|psi><psi|` on the chosen qubit and divides by the square root of the probability.
It is not unitary, so it acts on the statevector directly and returns
`(state, probability)`.

**Checked in the tests:**

- `'0'` and `'1'` against keeping only the matching amplitudes;
- `'+'`, `'-'`, `'+i'`, `'-i'` against the basis-change circuit (`h`, or `sdg` then
  `h`), postselection on `|0>` or `|1>`, and the inverse change;
- postselecting a qubit in the middle of a circuit against copying it with `cx` into a
  fresh ancilla and postselecting the ancilla at the end (Aaronson, Sect. 3).

::: dense_evolution.circuits.postselect
