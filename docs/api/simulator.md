# Simulator

`DenseSVSimulator` is the engine every other page's `sim.run_circuit(...)` call runs
against — a dense statevector of length `2**n`, held in memory and updated in place as
gates apply to it. Everything below is what that one object can do.

## Step 1. Create a simulator

```python
import dense_evolution as de

sim = de.DenseSVSimulator(2)
sim.n, sim.dim, sim.dtype
```

```
(2, 4, <class 'numpy.complex128'>)
```

`DenseSVSimulator` only needs one thing to start: how many qubits. `n` is that count,
`dim` is the statevector length `2**n`, `dtype` is `complex128` by default (Step 7 has
the `complex64` alternative). A fresh simulator starts in `|00>` — `sim.get_statevector()`
right now would show a `1` in the first entry and zero everywhere else.

## Step 2. Run a real circuit

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim.run_circuit(circuit.to_tuples())
sim.get_statevector().round(4)
```

```
array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])
```

`run_circuit` takes exactly the tuple list `QASMCircuit.to_tuples()` produces and applies
every gate to `sim`'s statevector in order, in place — `sim` is now the Bell state this
same circuit builds on every other page. `get_statevector()` reads the result back out as
a plain NumPy array.

## Step 3. Measurement probabilities

```python
sim.get_probabilities().round(4)
```

```
array([0.5, 0. , 0. , 0.5])
```

`get_probabilities()` is `|amplitude|^2` for every basis state, always real and
normalized to sum to 1 — the distribution a real device's repeated measurements would
approximate, as opposed to `get_statevector()`'s complex amplitudes, which no device
can read out directly.

## Step 4. Collapse a qubit with a real measurement

```python
import jax

sim2 = de.DenseSVSimulator(2)
sim2.run_circuit(circuit.to_tuples())
sim2.measure(0, jax_key=jax.random.PRNGKey(0))
```

```
1
```

`measure` is a real projective measurement, not a peek at `get_probabilities()`: it
samples one outcome from qubit 0's marginal, returns it, and collapses `sim2`'s
statevector to match — `sim2.get_statevector()` now has a single `1` at the `|11>` entry,
since this Bell state's qubits are always measured equal. `jax_key` makes the random
outcome reproducible; omit it for a genuinely random draw each call.

## Step 5. Why it's fast: the compiled path

```python
sim3 = de.DenseSVSimulator(2)
sim3.run_circuit_jit(circuit.to_tuples())
sim3.get_statevector().round(4)
```

```
array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])
```

Same circuit, same result — but `run_circuit_jit` compiles the whole circuit into one
XLA call instead of dispatching each gate from Python separately, measured 6x+ faster on
realistic circuits. `run_circuit` (Step 2) already calls this automatically whenever
every gate in the circuit is one it covers (see [Gates](gates.md)'s Details for exactly
which aren't); calling `run_circuit_jit` directly is only useful to force the compiled
path, or inside your own `jax.jit`-wrapped code.

## Step 6. Build a circuit one gate at a time

```python
sim4 = de.DenseSVSimulator(2)
sim4.apply_gate_1q(de.GATES['h'], 0)
sim4.apply_cx(0, 1)
sim4.get_statevector().round(4)
```

```
array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])
```

Identical result to Step 2, built without a circuit or a parser at all — this is what
`run_circuit` calls internally, one gate at a time. `apply_gate_1q`/`apply_gate_2q` take
any matrix from [`GATES`/`PARAMETRIC_GATES`](gates.md) directly; `apply_cx`, `apply_cz`,
`apply_rx`, `apply_ry`, `apply_rz` are named shortcuts for the most common ones. Useful
when a circuit is being constructed programmatically rather than written as QASM.

## Step 7. Memory and precision

```python
sim_small = de.DenseSVSimulator(20, use_float32=True)
sim_big = de.DenseSVSimulator(20, use_float32=False)
sim_small.memory_mb(), sim_big.memory_mb()
```

```
(8.388608, 16.777216)
```

`use_float32=True` stores the statevector as `complex64` instead of `complex128` — half
the memory, shown here on a 20-qubit register, at the cost of numerical precision. The
default (`complex128`) is what every other page on this site uses; switch to `complex64`
only once memory, not precision, is the bottleneck. For circuits too large for either
dtype to fit in memory at all, see [`Chunk`](chunk.md) instead.

---

## Details

### Qubit ordering is MSB-first

Qubit 0 is the *most* significant bit of the statevector index, not the least — the
opposite of some other simulators' convention. `physical_bit_position = n - 1 - qubit`
is the actual bit position `apply_gate_1q`/`measure`/etc. operate on internally; this
only matters if you're indexing into `get_statevector()`'s raw array yourself; every
gate/measurement method already takes qubit indices in the qubit-0-is-qubit-0 sense
this whole page uses.

### Starting from a custom state

`set_initial_state(state)` resets `sim` to any complex array of length `2**n` instead of
`|0...0>`, normalizing it automatically (`set_state` is the same method, aliased for the
VQE engine). Passing nothing (or explicitly `None`) resets back to `|0...0>`.

### `run_circuit`'s automatic fast-path dispatch

`run_circuit` takes the compiled path (Step 5) precisely when every gate name in the
circuit, after transpilation, has an entry in `GATE_IDS` — see [Gates](gates.md)'s
Details section for exactly which gates don't (`swap`, `ccx`, `ecr`, `iswap`, `u2`,
`u3`), and note that `QuantumTranspiler.transpile` (called automatically unless
`transpile=False`) already decomposes some of those into gates that do.

### Chunked execution for large, variable-length circuits

`run_circuit_with_chunking(circuit, chunk_size=500)` runs a circuit in fixed-size
chunks, each a separate `run_circuit_jit` call — XLA recompiles per distinct circuit
length, so a long circuit whose length varies run to run (e.g. a variational loop with
early stopping) recompiles far less often when split into same-sized pieces than when
run as one whole, differently-sized circuit each time.

### Batched execution over many parameter sets at once

`run_batch_jit(base_circuit, parameter_batch)` runs the same circuit shape once per row
of `parameter_batch` — one column per parametric gate, in the order it appears in
`base_circuit` — all `jax.vmap`'d together in a single compiled call, for e.g. evaluating
many VQE parameter guesses at once instead of one circuit run per guess.
`run_parametric_batch_jit` and `run_circuit_jit_beast_mode` are deprecated aliases for
`run_batch_jit`/`run_circuit_jit` kept for pre-8.1.46 code; both emit a
`DeprecationWarning` and will be removed in a future major version.

### An out-of-range qubit index on the compiled path doesn't raise the way you'd expect

`apply_gate_1q`/`apply_gate_2q`/`measure` all validate their qubit index directly and
raise `ValueError` immediately for one out of `[0, n)`. The compiled path (`run_circuit_jit`,
`run_batch_jit`) encodes qubit indices as bit-shift amounts inside `jax.lax.scan`/`switch`
instead, which does not raise on an out-of-range index — it silently corrupts the whole
statevector to all-zero. `_check_qubit_range` exists specifically to catch this before it
happens, on every code path that reaches the compiled kernels.

### `measure`'s bug history

The original NumPy branch zeroed the *wrong* basis-state slot on collapse (`result=0`
zeroed slot 1's amplitudes and vice versa) and never normalized the JAX branch at all — both
fixed. A second, separate bug affected only the JAX branch: it computed the moveaxis
target from `n - 1 - qubit_idx` (correct for the NumPy branch's raw stride arithmetic,
which is a genuinely different indexing scheme) instead of `qubit_idx` directly (what
the JAX branch's `reshape`-based indexing actually needs, matching `apply_gate_1q`'s own
convention) — silently measuring the wrong qubit's marginal whenever
`qubit_idx != n - 1 - qubit_idx`.

::: dense_evolution.backends.statevector

## See Also

- [Gates](gates.md) — the `GATES`/`PARAMETRIC_GATES`/`GATE_IDS` tables Step 6 and the
  fast-path dispatch (Step 5) both read from.
- [`QASMParser`](parser.md) — turns a QASM string into the tuples `run_circuit` expects.
- [`Chunk`](chunk.md) — anti-OOM slicing for circuits too large for one dense allocation
  at any dtype.
- [`MPSSimulator`](mps.md) — an alternative backend for low-entanglement circuits, at a
  much larger qubit count than a dense statevector can reach at all.
