# Chunk (large-scale, anti-OOM)

`DenseSVSimulator` allocates all `2**n` amplitudes the moment it's created — at 30
qubits that's already ~17 GB, before running a single gate. `Chunk` runs the same
kind of circuit without that up-front allocation, checking real available memory
first and splitting the statevector into RAM-sized pieces only when the qubit count
actually needs it.

## Step 1. Use it exactly like a simulator

```python
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];'
circuit = de.QASMParser().parse(qasm)

chunk = de.Chunk(4)
chunk.run_chunk(circuit.to_tuples())
chunk.get_probabilities().round(4)
```

```
array([0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5])
```

A 4-qubit GHZ chain — the standard `|0000>`/`|1111>` signature. `Chunk(n_qubits)`
and `run_chunk(circuit.to_tuples())` mirror `DenseSVSimulator`/`run_circuit_jit`
exactly; `get_probabilities()`/`get_statevector()` work the same way too. At a
small qubit count like this, `Chunk` checks that the array fits comfortably in
memory, then runs it as one ordinary `DenseSVSimulator` underneath — the chunking
machinery only activates once the qubit count actually needs it (Step 3).

## Step 2. See what `Chunk` actually decided

```python
chunk.num_chunks, chunk.chunk_size_bits
```

```
(1, 27)
```

`chunk_size_bits` is the largest qubit count `Chunk` judged safe to allocate as one
piece, computed from the *real* available memory on the machine running this code —
expect a different number on a different machine. `num_chunks == 1` here means 4
qubits fit entirely inside that safe budget, so nothing was actually split.

## Step 3. Why `Chunk` exists: the geometry at a qubit count that doesn't fit

```python
from dense_evolution.backends.chunk import MemoryChunker

geo = MemoryChunker(40)
geo.num_chunks, geo.chunk_dim, geo.chunk_size_bits
```

```
(8192, 134217728, 27)
```

`MemoryChunker` computes this geometry without allocating anything — pure
arithmetic, safe to call at any qubit count. A plain `DenseSVSimulator(40)` would
need `2**40` complex128 amplitudes, on the order of 17 TB, all at once. `Chunk(40)`
would instead hold 8192 separate pieces of `2**27` amplitudes each (~1 GB apiece,
`geo.memory_mb()`) — still a lot of pieces, but each one is a size real RAM can
actually hold, which is the entire point.

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
anything, and `CircuitChunker` (Step 1's single-chunk path) runs it again before
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

---

## Details

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
stacked representation — the same total element count as the separate chunks it
replaces, so the `(2**n_qubits,)`-shaped array this class exists to avoid is never
built. `SafeMemoryGuard.check_allocation` runs once up front, sized for
`num_chunks + 2` chunks held at once (headroom for the cross-chunk gate-mixing
math's own temporary arrays) — if that doesn't fit, construction fails immediately
with `MemoryPressureError`, before any of the `num_chunks` inner simulators are
allocated.

### No disk paging — this is *moderate* overflow, not unlimited

`Chunk` covers qubit counts that need more than one RAM-sized piece, as long as
all of those pieces still fit in RAM *at once* — there is no memmap/disk-backed
path for when even that doesn't fit. `SafeMemoryGuard.check_allocation` is what
catches this case and fails cleanly rather than letting the OS start swapping.

### `run_chunk_distributed`: one chunk per physical device

`run_chunk_distributed(circuit)` runs the same multi-chunk computation across a
real JAX device mesh — each chunk pinned to its own device, exchanging edge data
via `jax.lax.ppermute` instead of all chunks sharing one process's RAM. Requires
`jax.device_count() >= num_chunks`; raises `RuntimeError` immediately otherwise,
rather than silently falling back to the single-process path (which would quietly
give up the reason to call this method at all). Test with simulated multi-device
CPU via the `XLA_FLAGS=--xla_force_host_platform_device_count=N` environment
variable, set before the process starts (JAX's device count is fixed at first
initialization).

### `.sv` accepts an external statevector back

`chunk.sv = new_statevector` writes a full `(2**n,)` array back through to
whichever physical storage `Chunk` is actually using — split back into per-chunk
slices automatically when `num_chunks > 1`. This is what makes
`NoiseModel.apply_to_sv(chunk.sv, ...)` (see [Noise](noise.md)) work transparently
on a `Chunk` instance the same way it works on a plain `DenseSVSimulator`.

### Backward-compatible names

`chunk1`/`chunk2`/`Chunk2Incrociato` are aliases for `MemoryChunker`/`Chunk`/`Chunk`
respectively, kept for code written against this module's earlier internal names.

::: dense_evolution.chunk

## See Also

- [`DenseSVSimulator`](simulator.md) — the engine `Chunk` wraps; everything in
  Step 1 mirrors its own API one-for-one.
- [`MPSSimulator`](mps.md) — an alternative for large qubit counts that trades
  exactness for a bounded bond dimension instead of RAM-sized slicing.
- [Noise](noise.md) — `NoiseModel.apply_to_sv`, usable directly on `chunk.sv`.
