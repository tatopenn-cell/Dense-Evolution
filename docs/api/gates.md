# Gates

`dense_evolution.circuits.gates` is not a class or a function — it's three plain
dictionaries that every simulator backend looks a gate name up in when it runs a
circuit. Most code never imports this module directly: `QASMParser` already speaks
gate names, and `DenseSVSimulator.run_circuit` looks them up automatically. This page
is for the one time you want to see what a gate name actually resolves to.

## Step 1. A parsed circuit is a list of gate names

```python
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; rz(0.3) q[1]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
circuit.to_tuples()
```

```
[('h', 0), ('rz', 1, 0.3), ('cx', 0, 1)]
```

Each tuple starts with a gate name, followed by the qubit(s) it acts on and, if the
gate takes one, an angle. `h` and `cx` never need an angle — the same matrix every
time they're called. `rz` does — a different matrix for every angle. That split is
exactly the line between the two dictionaries below.

## Step 2. Fixed gates — `GATES`

```python
de.GATES['h'].round(4)
```

```
array([[ 0.7071+0.j,  0.7071+0.j],
       [ 0.7071+0.j, -0.7071+0.j]])
```

`GATES['h']` is the same matrix Step 1's `h q[0]` applies to qubit 0 — a plain NumPy
array, one entry per gate name, no angle involved. Single-qubit gates are 2×2
(`h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `sx`, `id`); two-qubit gates are 4×4
(`cx`, `cz`, `cy`, `swap`, `iswap`, `ecr`); `ccx` (Toffoli) is the only 3-qubit entry,
8×8.

## Step 3. Parametric gates — `PARAMETRIC_GATES`

```python
de.PARAMETRIC_GATES['rz'](0.3).round(4)
```

```
Array([[0.9888-0.1494j, 0.    +0.j    ],
       [0.    +0.j    , 0.9888+0.1494j]], dtype=complex128)
```

`PARAMETRIC_GATES['rz']` is a function, not a matrix — calling it with the same
`0.3` from Step 1's `rz(0.3) q[1]` builds the matrix that specific angle needs.
Every entry here works the same way: `rx`, `ry`, `rz`, `cp`, `crz` take one angle;
`u2` takes two (`phi`, `lam`); `u3` takes three (`theta`, `phi`, `lam`); `gphase`
takes one (`alpha`) and returns a global phase on the whole qubit, not a rotation.
`p`/`u1` are two names for the same one-angle gate.

## Step 4. Gate IDs — the fast-path lookup

```python
de.GATE_IDS['h'], de.GATE_IDS['cx'], de.GATE_IDS['rz']
```

```
(1, 20, 11)
```

`run_circuit` has two ways to execute a circuit: one Python call per gate, or —
when every gate in the circuit has an entry here — a single compiled JAX call for
the whole circuit at once, measured 6x+ faster. `GATE_IDS` is what that fast path
checks: the same three gates from Step 1, now as plain integers a JAX kernel can
switch on instead of a string it can't.

---

## Details

### Not every gate has a `GATE_IDS` entry

`swap`, `ccx`, `ecr`, `iswap`, `u2`, and `u3` have no entry in `GATE_IDS`, so a
circuit containing any of them always falls back to `run_circuit`'s eager,
one-gate-at-a-time path — `swap` in particular is never meant to reach this table
at all: `QuantumTranspiler.transpile` always decomposes it into three `cx` gates
first. `gphase` (id 14) is the exception that proves the rule: it isn't something
you write in QASM, only something `QuantumTranspiler.decompose_u3` emits when it
rewrites a `u2`/`u3` gate into `rz`/`ry`/`rz`/`gphase`.

### Aliases share one numeric ID

`p`, `u1`, and `phase` all map to `GATE_IDS` entry 12; `cp` and `cphase` both map to
22. Several QASM spellings for the same gate collapse to one kernel dispatch entry.

### Dispatching two-qubit parametric gates by name, not argument count

A one-qubit gate with two angles (`u2`: qubit, phi, lam) and a two-qubit gate with
one angle (`cp`/`crz`: q1, q2, theta) both produce a 3-element argument tuple, so
counting arguments can't tell them apart. `_TWO_QUBIT_PARAMETRIC_GATES`, a
`frozenset` of `{'cp', 'cphase', 'crz'}`, is what `run_circuit` actually checks —
by name, not by how many arguments came along with it.

### `GATES` is built with plain NumPy, on purpose

`GATES`'s entries are computed once, at import time, with NumPy rather than JAX.
Building them with JAX instead used to produce a real bug: if `jax_enable_x64`
wasn't already on at the exact moment this module was first imported, JAX silently
truncated `complex128` to `complex64` for every fixed gate, permanently — these are
module-level constants, never rebuilt later even after 64-bit precision turns on
for the actual simulation. That showed up as a measurable ~1e-8 unitarity violation
in circuits mixing a `complex128` statevector with these truncated matrices.
`PARAMETRIC_GATES`'s entries don't have this problem: they're lambdas, evaluated
lazily at gate-application time, by which point 64-bit precision has already been
turned on.

### An unknown gate name raises immediately

Both `run_circuit` and `run_circuit_jit` raise `ValueError` the moment they see a
name that's in neither `GATES`/`PARAMETRIC_GATES` nor `GATE_IDS`. A typo in a gate
name used to be silently dropped from the circuit instead (issue #4) — a wrong
circuit that ran without complaint is worse than one that stops immediately.

## See Also

- [`QASMParser`](parser.md) — turns a gate name in a QASM string into the tuples
  this page starts from.
- [`DenseSVSimulator`](simulator.md) — `run_circuit` and `run_circuit_jit`, the two
  dispatch paths Step 4 describes.
- [`QuantumTranspiler`](compiler.md) — decomposes `swap`/`u2`/`u3` into gates
  `GATE_IDS` actually covers, and is where `gphase` gets emitted.
