# Simulator

`DenseSVSimulator` is the engine every other page's `sim.run_circuit_jit(...)` call runs
against — a dense statevector of length `2**n`, held in memory and updated in place as
gates apply to it. This is a JAX library: the whole point of `DenseSVSimulator` is
running circuits as one compiled XLA call, not one Python step per gate — every example
below runs on that compiled path.

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
sim.run_circuit_jit(circuit.to_tuples())
sim.get_statevector().round(4)
```

```
array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])
```

`run_circuit_jit` takes exactly the tuple list `QASMCircuit.to_tuples()` produces,
compiles the whole circuit into one XLA call, and applies it to `sim`'s statevector in
place — `sim` is now the Bell state this same circuit builds on every other page.
`get_statevector()` reads the result back out as a plain NumPy array.

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
sim2.run_circuit_jit(circuit.to_tuples())
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

## Step 5. Build a circuit one gate at a time

```python
sim3 = de.DenseSVSimulator(2)
sim3.apply_gate_1q(de.GATES['h'], 0)
sim3.apply_cx(0, 1)
sim3.get_statevector().round(4)
```

```
array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])
```

Identical result to Step 2, built without a circuit or a parser at all. `apply_gate_1q`/
`apply_gate_2q` take any matrix from [`GATES`/`PARAMETRIC_GATES`](gates.md) directly;
`apply_cx`, `apply_cz`, `apply_rx`, `apply_ry`, `apply_rz` are named shortcuts for the
most common ones — each one still runs through the active backend (JAX when installed,
which is the normal case), just one gate at a time instead of a whole circuit compiled
at once. Useful when a circuit is being constructed programmatically rather than written
as QASM.

## Step 6. Run many parameter values at once

```python
import numpy as np

template = [('ry', 0, None)]
thetas = np.array([[0.0], [1.0], [2.0], [3.0]])
batch_sv = de.DenseSVSimulator(1).run_batch_jit(template, thetas)
(np.abs(np.asarray(batch_sv)) ** 2).round(4)
```

```
array([[1.    , 0.    ],
       [0.7702, 0.2298],
       [0.2919, 0.7081],
       [0.005 , 0.995 ]])
```

This is the actual reason a JAX-based simulator exists: `run_batch_jit` runs the *same*
circuit shape once per row of `thetas` — one column per parametric gate, `None` standing
in for "filled in from the batch" — all `jax.vmap`'d together as a single compiled call.
Row `i`'s output is exactly `RY(thetas[i])|0>`'s probabilities; a real VQE parameter
sweep is this same call with many more rows and a real circuit, still one call, not one
Python loop iteration per parameter set.

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

### `run_circuit` exists, but don't reach for it

`run_circuit` (plain, no `_jit` suffix) is an eager, one-gate-at-a-time Python loop —
the opposite of what this library is built around. It still ends up calling
`run_circuit_jit` internally on any real install (JAX is a core dependency, not
optional, so its own `HAS_JAX and all(gate in GATE_IDS ...)` check almost always passes),
but it pays for that with a redundant transpile-and-scan on every call that
`run_circuit_jit` skips by going straight there. It's kept only for the one case where
neither path can be taken — a gate name genuinely outside both `GATE_IDS` and
`GATES`/`PARAMETRIC_GATES` — and for backward compatibility with code written before
`run_circuit_jit` existed. Every example on this site calls `run_circuit_jit` (or
`run_batch_jit`) directly, on purpose.

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

### Chunked execution for large, variable-length circuits

`run_circuit_with_chunking(circuit, chunk_size=500)` runs a circuit in fixed-size
chunks, each a separate `run_circuit_jit` call — XLA recompiles per distinct circuit
length, so a long circuit whose length varies run to run (e.g. a variational loop with
early stopping) recompiles far less often when split into same-sized pieces than when
run as one whole, differently-sized circuit each time.

### `run_batch_jit`'s deprecated alias

`run_parametric_batch_jit` and `run_circuit_jit_beast_mode` are deprecated aliases for
`run_batch_jit`/`run_circuit_jit`, kept for pre-8.1.46 code; both emit a
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

- [Gates](gates.md) — the `GATES`/`PARAMETRIC_GATES`/`GATE_IDS` tables Step 5 and
  `run_circuit_jit`'s dispatch both read from.
- [`QASMParser`](parser.md) — turns a QASM string into the tuples `run_circuit_jit`
  expects.
- [`Chunk`](chunk.md) — anti-OOM slicing for circuits too large for one dense allocation
  at any dtype.
- [`MPSSimulator`](mps.md) — an alternative backend for low-entanglement circuits, at a
  much larger qubit count than a dense statevector can reach at all.
