# QASM Parser

`QASMParser` turns an OpenQASM 2.0 or 3.0 string into a `QASMCircuit` — the object every
other page's `circuit.to_tuples()` call feeds straight into `DenseSVSimulator.run_circuit`.
This page covers what QASM the parser actually understands.

## Step 1. Parse your first circuit

```python
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
circuit.n_qubits, circuit.to_tuples()
```

```
(2, [('h', 0), ('cx', 0, 1)])
```

`parse` reads the register declaration (`qreg q[2]`) and every gate statement, and
returns a `QASMCircuit` — `n_qubits` is read straight from the register, `to_tuples()`
is the tuple list every simulator page on this site starts from.

## Step 2. Registers, and bare register names

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; h q; x q[2];'
circuit = de.QASMParser().parse(qasm)
circuit.to_tuples()
```

```
[('h', 0), ('x', 2)]
```

`q[2]` is qubit 2, exactly as written. `q` with no index — legal OpenQASM, meaning "the
whole register" in some contexts — resolves here to qubit 0 of that register, not every
qubit in it; write out `q[0]`, `q[1]`, `q[2]` explicitly if that's what you mean.

## Step 3. Parametric gates: real math, not just numbers

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; rx(pi/2) q[0]; ry(sqrt(2)) q[0];'
circuit = de.QASMParser().parse(qasm)
[(op['name'], round(op['params'][0], 4)) for op in circuit.ops]
```

```
[('rx', 1.5708), ('ry', 1.4142)]
```

A gate parameter isn't limited to a literal number — `pi`, `tau`, `euler`, and functions
like `sqrt`, `sin`, `cos`, `exp`, `log` all evaluate exactly as written, including
combined expressions like `pi/2`. `circuit.ops` (the pre-tuple form `to_tuples()` builds
from) is where the evaluated float ends up, one per gate.

## Step 4. Range syntax

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0:3];'
circuit = de.QASMParser().parse(qasm)
circuit.to_tuples()
```

```
[('h', 0), ('h', 1), ('h', 2)]
```

`q[0:3]` means qubits 0 through 2 — exclusive of the upper bound, the same convention
Python's own slicing uses. Applying a single-qubit gate like `h` to a range expands into
one gate application per qubit, not one gate call carrying three qubits at once.

## Step 5. Gate name aliases

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; cnot q[0],q[1]; toffoli q[0],q[1],q[0];'
circuit = de.QASMParser().parse(qasm)
[op['name'] for op in circuit.ops]
```

```
['cx', 'ccx']
```

`cnot` and `toffoli` aren't gate names this library's simulator knows about — they're
common alternate names some QASM sources use for `cx` and `ccx`. The parser normalizes
these (and `cu1`→`cp`, `u1`→`p`, `fredkin`→`cswap`) before a gate name ever reaches
`GATES`/`PARAMETRIC_GATES`, so both spellings run identically.

## Step 6. OpenQASM 3.0

```python
qasm = 'OPENQASM 3.0; qubit[3] q; bit[3] c; for int i in [0:2] { h q[i]; }'
circuit = de.QASMParser().parse(qasm)
circuit.to_tuples()
```

```
[('h', 0), ('h', 1), ('h', 2)]
```

`qubit[3] q` and `bit[3] c` are QASM 3.0's register syntax — `qreg`/`creg`'s replacement.
`for int i in [0:2] { ... }` is unrolled before parsing proper begins, substituting `0`,
`1`, `2` into the loop body in turn — QASM 3.0's `for` range is inclusive of its upper
bound, unlike Step 4's `q[a:b]` qubit-range syntax, which is exclusive.

## Step 7. Validate before running

```python
from dense_evolution.circuits.parser import QASMCircuit

bad = QASMCircuit(n_qubits=2, n_cbits=0,
                   ops=[{'type': 'gate', 'name': 'h', 'qubits': [5], 'params': []}])
de.QASMParser().validate(bad)
```

```
(False, "Gate 'h' at op[0] references qubit 5 but n_qubits=2.")
```

`validate` is a light structural check — no `qreg`/gate-count mismatch, no qubit index
outside `[0, n_qubits)` — meant for a `QASMCircuit` you built or edited by hand rather
than one straight out of `parse` (a genuinely parsed circuit's `n_qubits` already
accounts for every qubit index its own gates reference). It does not check gate
semantics — a circuit using a gate name the simulator doesn't recognize still passes.

## Step 8. Skip `to_tuples()` — a `QASMCircuit` is already iterable

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
list(circuit)
```

```
[('h', 0), ('cx', 0, 1)]
```

Every earlier step called `circuit.to_tuples()` explicitly, but `QASMCircuit` also
implements `__iter__` over that same tuple list — passing `circuit` itself anywhere a
plain tuple list is expected (`DenseSVSimulator.run_circuit`, `QuantumTranspiler.transpile`,
`Chunk.run_chunk`) works without the extra call.

---

## Details

### The parameter evaluator is a safe AST whitelist, not `eval()`

Gate parameters like Step 3's `sqrt(2)` used to be evaluated with
`eval(tok, {'__builtins__': {}})` — this does **not** stop attribute/dunder traversal of
the live object graph: `().__class__.__bases__[0].__subclasses__()` needs no builtin
name at all, and from there any class already loaded in the process is reachable. Passed
as a gate parameter through the public `parse()` entry point, that expression executed
successfully before this fix. The evaluator now walks the parameter's AST against an
explicit whitelist (numeric literals, `+-*/%**`, unary `+`/`-`, and name/call lookups
restricted to a fixed math environment) — an `ast.Attribute` node is never one of the
handled cases, so any expression containing `.` always falls through to rejection,
structurally rather than by pattern-matching dangerous names.

### A malformed parameter expression raises, not silently becomes `0.0`

An unparseable or disallowed parameter expression (a typo, or the injection attempt
above) raises `ValueError` immediately. It used to fall back to `0.0` silently — turning
a typo into a different, valid-looking circuit (`rx(0)` instead of the intended angle)
with no signal anything was wrong, the same class of silent-wrong-behavior this
codebase's gate-name and parameter-batch validation elsewhere were also fixed to avoid.

### Comments and unresolved control flow are stripped, not executed

`/* block */` and `// line` comments are removed before parsing. `if`/`while`/`def`
blocks, and any `for` loop whose bounds aren't a literal or a previously-declared
`int`/`const int` variable (Step 6 needs resolvable bounds to unroll), are stripped
entirely rather than run — there's no runtime classical-bit state to execute them
against. A `gate NAME(...) { ... }` definition (common in Qiskit's OpenQASM 2.0 export
for composite gates) is stripped the same way; a later call to that gate name still
falls through as an unrecognized gate, same as any other unknown name.

### `to_tuples()`'s field order, and why `QASMCircuit` is iterable at all

`to_tuples()` returns `(name, *qubits, *params)` — qubits always immediately after the
name, params trailing. `__iter__` (Step 8) exists because `QuantumTranspiler.transpile`,
reached via `Chunk.run_chunk(circuit)`, iterates its `circuit` argument directly — handed
a `QASMCircuit` straight from `parse()` instead of `circuit.to_tuples()`, that used to
raise `TypeError: 'QASMCircuit' object is not iterable`.

::: dense_evolution.circuits.parser

## See Also

- [`DenseSVSimulator`](simulator.md) — `run_circuit` is what every `to_tuples()` output
  on this page is built to feed.
- [Gates](gates.md) — the `GATES`/`PARAMETRIC_GATES` tables a parsed gate name resolves
  against once it reaches the simulator.
- [`QuantumTranspiler`](compiler.md) — runs on a `QASMCircuit`'s tuples before the
  simulator ever sees them, decomposing anything it can't execute directly.
