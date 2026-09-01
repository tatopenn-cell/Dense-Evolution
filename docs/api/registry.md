# Registry (hardware detection)

Before running a big circuit, it's worth knowing how big "big" safely is on the actual
machine running it -- `QuantumHardwareRegistry` reads the current machine's RAM and GPU
availability and suggests a qubit ceiling from that, once, at construction time.
Despite living in `dense_evolution.circuits.registry` historically, this has nothing to
do with noise -- see [Noise](noise.md) for `NoiseModel`/`NoiseSpec` instead.

## Step 1. What does this machine look like?

```python
import dense_evolution as de

reg = de.QuantumHardwareRegistry()
reg.ram_total, reg.has_jax, reg.has_gpu, reg.max_dense_qubits
```

```
(7.877658843994141, True, False, 20)
```

`ram_total` is total system RAM in GB (this machine's own, whatever it happens to be),
`has_jax`/`has_gpu` are booleans, and `max_dense_qubits` is a suggested ceiling for a
dense statevector simulation: `28` at `ram_total >= 50`, `24` at `>= 12`, `20`
otherwise -- three fixed tiers, not a formula fit to this machine's exact number. `20`
above reflects an 8GB machine landing in the lowest tier.

## Step 2. The same numbers, printed

```python
reg.print_diagnostics()
```

```
MAX_DENSE=20q | JAX=True | GPU=False
```

`print_diagnostics()` is the same four fields from Step 1, condensed to one line --
useful as a quick sanity check at the top of a script before committing to a large
qubit count.

---

## Details

**`max_dense_qubits` is a suggestion, not an enforced limit**: nothing in this class
stops a caller from constructing a `DenseSVSimulator` above it -- pair it with
[`Chunk`](chunk.md)'s `SafeMemoryGuard`, which does actively refuse an allocation once
available memory drops below its own threshold, for a real enforced ceiling instead of
an advisory one.

**Lazy `x64`**: constructing `QuantumHardwareRegistry` is one of the entry points that
enables `jax_enable_x64` the first time it runs, same as `DenseSVSimulator`/
`circuit_to_energy_fn` -- see [Autodiff](autodiff.md)'s own precision note.

::: dense_evolution.circuits.registry
