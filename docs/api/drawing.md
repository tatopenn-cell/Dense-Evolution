# Drawing (plain-text circuit diagrams)

The fast way to sanity-check what a gate-tuple list actually builds, without running
it or leaving the terminal -- a plain-text diagram, deliberately ASCII-only (`-`, `|`,
`*`), not Unicode box-drawing, so a printed diagram survives any console encoding.

## Step 1. A circuit, as text

```python
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
print(de.draw_circuit(circuit.to_tuples(), n_qubits=2))
```

```
q0: --H----*--
q1: -------X--
```

`draw_circuit(circuit, n_qubits)` reads the same gate-tuple list every other page's
`run_circuit_jit` runs, one wire per qubit, gates in the order they appear left to
right -- `H` on `q0`, then `cx` linking `q0` (`*`, the control) to `q1` (`X`, the
target). Nothing to render or save -- the return value is a plain Python string,
printable anywhere.

---

## Details

**Not the same as `plot_circuit`**: [`Diagram`](diagram.md)'s `plot_circuit` draws the
same circuit as an actual saved figure (Quirk-style colored boxes) instead of a
terminal string -- reach for that one when a real image is needed (a docs page, a
report), this one for a quick terminal check.

::: dense_evolution.utils.drawing
