# Chunk (large-scale, anti-OOM)

`DenseSVSimulator` allocates all `2**n` amplitudes the moment it's created — at 30
qubits that's already ~17 GB, before running a single gate. `Chunk` runs the same
kind of circuit without that up-front allocation: it reads the *actual* compute
device's free memory (VRAM on GPU/TPU, not just host RAM — Step 4's Details section
has the real Colab OOM bug that made this distinction necessary), decides how many
RAM-sized pieces the circuit needs right now, and runs gates across those pieces
with a compiled kernel that never builds the full array — including, when more than
one physical device is available, spreading the pieces across a real device mesh
instead of one process's RAM (Step 6). None of that shows up in the
public API: `Chunk` looks exactly like `DenseSVSimulator` to call.

## The pieces, and how they fit together

Five names live in this module. You only ever call one of them directly — the rest
exist to make that one work.

| Name | What it actually is | Do you call it? |
| :--- | :--- | :--- |
| **`Chunk`** | The public class. Decides everything below automatically. | **Yes — this is the only one most code needs.** |
| `MemoryChunker` | Pure arithmetic: given `n_qubits` and the machine's real free memory *right now*, computes how many RAM-sized pieces are needed. Allocates nothing. | Only to inspect the geometry without building a `Chunk`. |
| `SafeMemoryGuard` | Checks real RAM/VRAM before an allocation and raises `MemoryPressureError` if it isn't safe. | Only to check memory status yourself, outside a `Chunk`. |
| `CircuitChunker` | Runs a circuit in gate-sized slices against *one* simulator, so XLA doesn't recompile a new trace shape every call. | Never directly — `Chunk` owns one internally. |
| `MemoryPressureError` | The exception `SafeMemoryGuard` raises. | To catch it, if you want to react instead of crashing. |

`Chunk(n_qubits)` always does the same two things: ask `MemoryChunker` how many
pieces `n_qubits` needs *on this machine, right now*, then either (a) — fits in one
piece — hand the circuit to a `CircuitChunker` wrapping one ordinary
`DenseSVSimulator`, or (b) — needs more than one piece — hold that many simulators
at once and run gates across them with a compiled kernel that never builds the full
`(2**n,)` array. Step 1 below is case (a); Step 2 is case (b).

## Step 1. The common case: one circuit, one piece

```python
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];'
circuit = de.QASMParser().parse(qasm)

chunk = de.Chunk(4)
chunk.run_chunk(circuit.to_tuples())
chunk.num_chunks, chunk.get_probabilities().round(4)
```

```
(1, array([0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5]))
```

A 4-qubit GHZ chain — the standard `|0000>`/`|1111>` signature.
`num_chunks == 1`: on a real machine, 4 qubits fit comfortably inside the safe
budget (`chunk.chunk_size_bits`, usually well above 20 — see Step 3), so `Chunk`
just wraps one ordinary `DenseSVSimulator` and behaves identically to it —
`run_chunk`/`get_probabilities`/`get_statevector` mirror
`run_circuit_jit`/`get_probabilities`/`get_statevector` one-for-one. This is what
almost every real circuit does; the actual splitting in Step 2 only activates once
`n_qubits` genuinely exceeds what fits in memory.

## Step 2. Forcing the split, to see it for real

On a real machine, `num_chunks > 1` only kicks in at a qubit count too large to
demonstrate directly in a doc (dozens of qubits, gigabytes per piece). To show the
actual multi-piece code path deterministically regardless of this machine's real
RAM, override `get_dynamic_chunk` — the same technique this project's own test
suite uses to test the split cheaply.

```python
de.chunk.get_dynamic_chunk = lambda dtype_target: 2

chunk2 = de.Chunk(4)
chunk2.num_chunks, chunk2.chunk_size_bits
```

```
(4, 2)
```

Capping the safe budget at 2 qubits forces the same 4-qubit request from Step 1 to
split into 4 separate 2-qubit pieces instead of 1. Running the identical circuit now
exercises the real cross-piece kernel — `cx q[1],q[2]` connects two *different*
pieces, not two qubits inside the same array:

```python
chunk2.run_chunk(circuit.to_tuples())
chunk2.get_probabilities().round(4)
```

```
array([0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5])
```

Identical result to Step 1 — same circuit, same physics — but this time computed as
4 separate `(4,)`-element pieces held in RAM together, never as one `(16,)`-element
array. `get_dynamic_chunk` is only overridden here to make the split happen at a
size small enough to show; real code never touches it — `Chunk` calls it
automatically, sized to whatever machine it's actually running on.

## Step 3. Why this matters: the geometry at a qubit count that doesn't fit

```python
from dense_evolution.backends.chunk import MemoryChunker

geo = MemoryChunker(40)
geo.num_chunks, geo.chunk_dim, geo.chunk_size_bits
```

```
(8192, 134217728, 27)
```

`MemoryChunker` computes this geometry without allocating anything — pure
arithmetic, safe to call at any qubit count, on any machine (expect different
numbers on a different machine; this reflects real available memory here and now).
A plain `DenseSVSimulator(40)` would need `2**40` complex128 amplitudes, on the
order of 17 TB, all at once. `Chunk(40)` would instead hold 8192 separate pieces of
`2**27` amplitudes each (~1 GB apiece, `geo.memory_mb()`) — the same mechanism Step
2 just showed at a size you can actually run.

## Step 4. The anti-OOM guard blocks unsafe allocations before they happen

```python
from dense_evolution.backends.chunk import SafeMemoryGuard, MemoryPressureError

guard = SafeMemoryGuard(threshold_pct=0.99)
try:
    guard.check("demo")
except MemoryPressureError:
    print("blocked before any allocation was attempted")
```

```
blocked before any allocation was attempted
```

`threshold_pct=0.99` demands 99% of memory free after the check — essentially
never true on a real machine, so this always raises. `Chunk.__init__` runs the
same check (a realistic `threshold_pct`, default 0.15) *before* allocating
anything, and `CircuitChunker` (Step 1's single-piece path) runs it again before
every gate-slice during execution — the failure mode this exists for is
`jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED` crashing the whole
process partway through a run; `MemoryPressureError` raised up front, with a
clear message, is the alternative.

## Step 5. Diagnostics

```python
repr(chunk)
```

```
"Chunk(n_qubits=4, safe_qubits=4, num_chunks=1, chunk_size_bits=27, dtype=<class 'jax.numpy.complex128'>, mem_per_chunk=0.0 MB, ram_free=38.9%, has_jax=True)"
```

`ram_free` reflects the machine's real state at the moment `repr` is called, not a
cached value from construction — expect a different number here too; useful to
check right before a large run, not just after a failure.

## Step 6. Beyond one process: real multiple devices

Everything above holds all `num_chunks` pieces in one process's RAM. When more than
one physical JAX device is actually available, `run_chunk_distributed` runs the
identical computation with each piece pinned to its own device instead — real
point-to-point network exchange (`jax.lax.ppermute`) between devices for gates that
connect two pieces, not a bigger single-machine array.

```python
try:
    chunk2.run_chunk_distributed(circuit.to_tuples())
except RuntimeError as e:
    str(e)
```

```
'dispatch_distributed() needs >= 4 JAX devices (one per chunk), only 1 available. Force extra CPU devices for testing via the XLA_FLAGS environment variable: --xla_force_host_platform_device_count=N ...'
```

This machine has one JAX device, so `chunk2`'s 4 pieces (Step 2) can't spread across
4 real ones — `run_chunk_distributed` raises immediately with a clear, specific
error rather than silently running the single-process path instead, which would
quietly give up the reason to call it at all. Force extra CPU devices to actually
exercise this path locally: set `XLA_FLAGS=--xla_force_host_platform_device_count=N`
*before the process starts* (JAX's device count is fixed at first initialization).

---

## Details

### Two different reasons a circuit gets sliced

`Chunk` uses "chunking" for two genuinely different things, easy to conflate:
gate-slicing (`CircuitChunker`, Step 1's path — the *statevector* is one array,
but the *circuit* is split into `chunk_size_gates`-sized pieces purely so XLA
doesn't recompile a fresh trace shape on every call) versus statevector-splitting
(Step 2's path — the *array itself* is too large for one piece, so it's held as
several smaller arrays instead). A circuit only ever goes through one path, decided
once, in `Chunk.__init__`, by whether `n_qubits` fits `chunk_size_bits`.

### Building a second `Chunk` with the same geometry doesn't recompile

`num_chunks > 1`'s kernel (`_build_multi_chunk_runner`/`_build_distributed_chunk_runner`)
is memoized on `(num_chunks, m, k)` — the three plain integers that fully determine
it. Without this, every `Chunk.__init__` built a brand-new Python closure and wrapped
it in a fresh `jax.jit`, so two `Chunk` instances with identical geometry could never
hit JAX's own compilation cache (that cache is keyed by the wrapped function's
identity, not by structural equality of what it captured) — each one silently repaid
the full XLA compile cost the other had already paid. Measured directly: a second
`Chunk(4)` built right after a first one with the same forced geometry went from
0.654s (cold compile) to 0.123s (cache hit, verified same runner object,
`c1._multi_chunk_runner is c2._multi_chunk_runner`) — this is what actually makes
repeated `Chunk` construction (a VQE loop, a parameter sweep, this test suite's own
90+ `Chunk(...)` calls) cheap after the first one, automatically, with nothing for a
caller to do differently.

### Sizing from device memory, not just host RAM

The safe qubit budget (`get_dynamic_chunk`, behind `chunk_size_bits`) reads the
*active compute device's* own memory via `jax.devices()[0].memory_stats()` when
available, falling back to host RAM (`psutil`) only when it isn't (e.g. plain
CPU). This matters concretely on GPU: sizing chunks off host RAM while the actual
data lives in a much smaller GPU VRAM pool caused real `MemoryPressureError`/OOM
crashes well before VRAM was actually exhausted — verified on a real Colab T4 GPU
before this was fixed, where chunk sizing used >11 GB of host RAM headroom while
the data lived in the T4's 11-15 GB VRAM.

### `num_chunks > 1`: no statevector is ever fully materialized

Above the safe budget, `Chunk` holds `num_chunks` separate chunk-sized
`DenseSVSimulator` instances instead of one giant array, and `run_chunk` dispatches
gates against them via one compiled `jax.lax.scan` over a `(num_chunks, chunk_dim)`
stacked representation — the same total element count as the separate pieces it
replaces, so the `(2**n_qubits,)`-shaped array this class exists to avoid is never
built. `SafeMemoryGuard.check_allocation` runs once up front, sized for
`num_chunks + 2` pieces held at once (headroom for the cross-piece gate-mixing
math's own temporary arrays) — if that doesn't fit, construction fails immediately
with `MemoryPressureError`, before any of the `num_chunks` inner simulators are
allocated.

### No disk paging — this is *moderate* overflow, not unlimited

`Chunk` covers qubit counts that need more than one RAM-sized piece, as long as
all of those pieces still fit in RAM *at once* — there is no memmap/disk-backed
path for when even that doesn't fit. `SafeMemoryGuard.check_allocation` is what
catches this case and fails cleanly rather than letting the OS start swapping.

### `run_chunk_distributed`'s one-piece-per-device scope

Step 6's distributed path currently supports exactly one piece per physical
device (`jax.device_count() >= num_chunks` required) — the literal v1 reading of
the feature request behind it ("spread pieces across a device mesh"), not a
hybrid scheme with several pieces sharing one device, which remains a possible
future refinement. `jax.lax.ppermute`'s communication topology (`perm=`) must be
static — known at trace time — so it can't be built from the traced `q1`/`q2`
qubit indices directly; every possible chunk-select stride is instead enumerated
as its own statically-built `ppermute` call ahead of time, and `jax.lax.switch`
picks the right one at runtime.

### `.sv` accepts an external statevector back

`chunk.sv = new_statevector` writes a full `(2**n,)` array back through to
whichever physical storage `Chunk` is actually using — split back into per-piece
slices automatically when `num_chunks > 1`. This is what makes
`NoiseModel.apply_to_sv(chunk.sv, ...)` (see [Noise](noise.md)) work transparently
on a `Chunk` instance the same way it works on a plain `DenseSVSimulator`.

### Backward-compatible names

`chunk1`/`chunk2`/`Chunk2Incrociato` are aliases for `MemoryChunker`/`Chunk`/`Chunk`
respectively, kept for code written against this module's earlier internal names —
unrelated to Step 2's `chunk2` variable name, a coincidence of this page's own
narrative, not the alias.

::: dense_evolution.chunk

## See Also

- [`DenseSVSimulator`](simulator.md) — the engine `Chunk` wraps; Step 1 mirrors its
  own API one-for-one.
- [`MPSSimulator`](mps.md) — an alternative for large qubit counts that trades
  exactness for a bounded bond dimension instead of RAM-sized slicing.
- [Noise](noise.md) — `NoiseModel.apply_to_sv`, usable directly on `chunk.sv`.
