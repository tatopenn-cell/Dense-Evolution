# MPS Simulator

`DenseSVSimulator` keeps every one of a circuit's `2**n` amplitudes in memory — that
works for a couple dozen qubits, but beyond that it runs out of RAM.
`MPSSimulator` runs the same circuits at hundreds of qubits by keeping a
compact tensor-network representation instead of the full state, at the cost of
accuracy on highly-entangled circuits.

> **When to use MPS:** low-entanglement circuits — GHZ chains, shallow local
> circuits, product-state preparations. For highly-entangled circuits the bond
> dimension grows exponentially and MPS degrades back toward the same cost as
> the dense engine. It is a complement to `DenseSVSimulator`, not a replacement.

---

## Step 1. Build a circuit

A GHZ chain: qubit 0 goes into superposition, then every qubit is entangled with
the next one. This is exactly the kind of circuit MPS is built for — the
entanglement stays low no matter how many qubits the chain grows to.

```python
import dense_evolution as de

n = 50
lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
         f"qreg q[{n}];", f"creg c[{n}];", "h q[0];"]
lines += [f"cx q[{i}],q[{i+1}];" for i in range(n - 1)]
lines.append("measure q -> c;")

circuit = de.QASMParser().parse("\n".join(lines))
```

50 qubits is written with a loop instead of by hand, but the result is the same
real OpenQASM 2.0 text the parser always takes.

---

## Step 2. Run it with MPS instead of the full state

A dense statevector at 50 qubits would require `2**50` complex numbers — roughly
16 PB — far more memory than any machine has. `MPSSimulator` keeps a bounded
**bond dimension** (`max_bond`) instead, which is enough for a circuit this simple.

```python
mps = de.MPSSimulator(n_qubits=n, max_bond=16)
mps.run_circuit_jit(circuit.to_tuples())
```

`run_circuit_jit` compiles the whole circuit into a single `jax.lax.scan`-fused,
`@jax.jit`-compiled kernel. Use the eager per-gate methods (`apply_gate_1q`,
`apply_gate_2q`) instead when memory is the priority over speed — the JIT path
pads every gamma/lambda tensor to `max_bond` for the lifetime of the instance.

---

## Step 3. Read out the result

`get_top_k_probable_states` finds the most likely outcomes via greedy beam search,
without ever materialising a `2**50`-sized array.

```python
idx, probs = mps.get_top_k_probable_states(k=2)
for i, p in zip(idx, probs):
    print(f"{i:0{n}b}: {p:.4f}")
```

```
00000000000000000000000000000000000000000000000000: 0.5000
11111111111111111111111111111111111111111111111111: 0.5000
```

Only two outcomes are populated — all-zeros and all-ones, each at probability 0.5.
That is the GHZ signature; everything else is numerically zero.

> **Note:** beam search recall is not guaranteed for any fixed `k`. If a known
> state is missing from the output, increase `k` (e.g. `k=128`).

---

## Step 4. Check accuracy

```python
print(mps.summary())
```

```
MPSSimulator | n=50 | chi_max=16 | chi_used=16 | mem=0.198MB | trunc_err=1.61e-06 | avg_JSD=0.0000 | EE_max=1.000b | budget_violations=0
```

| Field | Meaning |
| :--- | :--- |
| `chi_max` | Hard cap on bond dimension (`max_bond` argument) |
| `chi_used` | Largest bond dimension actually reached during the run |
| `trunc_err` | Cumulative singular-value truncation error |
| `avg_JSD` | Mean Jensen-Shannon distance between full and truncated singular-value distributions across all SVD steps |
| `EE_max` | Peak entanglement entropy across all bonds (bits) |
| `budget_violations` | Times `max_bond` was hit before `jsd_budget` could be satisfied — if > 0, raise `max_bond` |

Both `trunc_err` and `avg_JSD` are near zero here. `budget_violations=0` means
the accuracy budget was never hit — the result is reliable at this `max_bond`.

> **Tip:** enable `jax_enable_x64` before importing the package for full
> `float64`/`complex128` precision. On this circuit it drops `trunc_err` from
> `1.61e-06` to `3.75e-15` and reduces `chi_used` from 16 to 2.
>
> ```python
> import jax
> jax.config.update("jax_enable_x64", True)
> import dense_evolution as de
> ```

---

## Details

### Troubleshooting

| Problem | Likely cause | Fix |
| :--- | :--- | :--- |
| `budget_violations > 0` or `avg_JSD` is high | `max_bond` is too small: the bond-dimension search hits the cap before `jsd_budget` is satisfied | Increase `max_bond` (e.g. `16 → 64`). Raising `jsd_budget` only loosens the tolerance — it does not improve accuracy. |
| `contract_to_statevector` raises `MemoryError` | `n > 24` is a hard cutoff, not a guideline | Use `get_probabilities_sampled` or `get_top_k_probable_states` instead — neither materialises a `(2**n,)` array. |
| `get_top_k_probable_states` misses a known state | Greedy beam search recall is not guaranteed for a fixed `k` | Increase `k` (e.g. `32 → 128`). |
| Simulation is slow or memory is high after `run_circuit_jit` | Tensors are padded to `max_bond` for the instance lifetime | Reduce `max_bond`, or use the eager per-gate path for short, low-entanglement circuits. |

### Performance

`DenseSVSimulator` and `Chunk` both require `2**n × 16` bytes for the
statevector alone — around 17 GB at 30 qubits, before any computation.
`MPSSimulator` scales as `O(n × max_bond²)` instead, so a low-entanglement
circuit like the GHZ chain above runs at 50+ qubits on an ordinary machine where
the dense engines would raise `MemoryError`.

Exact wall-clock numbers depend heavily on the machine, circuit depth, and
entanglement structure, so none are quoted here as a general result. Run
`mps.summary()` to see the numbers for your own machine and circuit, the same
way Step 4 above does.

---

::: dense_evolution.backends.mps

**See also:** [`DenseSVSimulator`](simulator.md) for exact statevector simulation
when entanglement is too high for a bounded bond dimension, and [`Chunk`](chunk.md)
for anti-OOM dense simulation at large qubit counts without bond-dimension
truncation.
