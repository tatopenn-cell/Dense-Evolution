# MPS Simulator

`DenseSVSimulator` keeps every one of a circuit's `2**n` amplitudes in memory. That
works for a couple dozen qubits — beyond that it runs out of RAM. `MPSSimulator` runs
the same kind of circuit at hundreds of qubits instead, as long as the circuit doesn't
get too entangled.

## Step 1. Build a circuit

A GHZ chain: qubit 0 goes into superposition, then every qubit is entangled with the
next one. This is exactly the kind of circuit MPS is built for — the entanglement stays
low no matter how many qubits it grows to.

```python
n = 50
lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];", "h q[0];"]
lines += [f"cx q[{i}],q[{i + 1}];" for i in range(n - 1)]
lines.append("measure q -> c;")
qasm = "\n".join(lines)

import dense_evolution as de

circuit = de.QASMParser().parse(qasm)
```

50 qubits is written with a loop instead of by hand, but the result is the same real
OpenQASM text the parser always takes.

## Step 2. Run it with MPS instead of the full state

A dense statevector at 50 qubits would need `2**50` complex numbers — far more memory
than any machine has. `MPSSimulator` keeps a bounded "bond dimension" (`max_bond`)
instead of the full state, which is enough for a circuit this simple.

```python
mps = de.MPSSimulator(n_qubits=n, max_bond=16)
mps.run_circuit_jit(circuit.to_tuples())
```

## Step 3. Read out the result

`get_top_k_probable_states` finds the most likely outcomes without ever building a
`2**50`-sized array.

```python
idx, probs = mps.get_top_k_probable_states(k=2)
for i, p in zip(idx, probs):
    print(f"{i:0{n}b}: {p:.4f}")
```

```
00000000000000000000000000000000000000000000000000: 0.5000
11111111111111111111111111111111111111111111111111: 0.5000
```

Only two outcomes are populated: all-zeros and all-ones, each with probability 0.5 —
the GHZ signature. Everything else is (numerically) zero.

## Step 4. Check how much was traded away for that scale

```python
print(mps.summary())
```

```
MPSSimulator | n=50 | chi_max=16 | chi_used=16 | mem=0.198MB | trunc_err=1.61e-06 | avg_JSD=0.0000 | EE_max=1.000b | budget_violations=0
```

`trunc_err` and `avg_JSD` are how far the truncated result is from the exact one — both
near zero here. `budget_violations` counts how many times `max_bond` had to cut more
than the accuracy budget allowed; if it's above 0, raise `max_bond`.

---

## Details

### Troubleshooting

| Problem | Likely cause | Fix |
| :--- | :--- | :--- |
| `budget_violations > 0` or `avg_JSD` is high | `max_bond` too small for this circuit's entanglement: `budget_violations` increments exactly when the achieved JSD exceeds `jsd_budget` because `max_bond` capped the bond first | Increase `max_bond` (e.g. 16 -> 64). Raising `jsd_budget` instead only loosens the tolerance, it does not improve accuracy. |
| `contract_to_statevector` raises `MemoryError` | `n > 24` is a hard, exact cutoff, not a soft guideline | Use `get_probabilities_sampled` or `get_top_k_probable_states` instead |
| `get_top_k_probable_states` misses a known state | Greedy beam search recall is not guaranteed for a fixed `k` | Increase `k` (e.g. 32 -> 128) |
| Simulation is slow or memory usage is high | Gamma/lambda tensors are padded to `max_bond` for every circuit, even a low-entanglement one | Reduce `max_bond`, or use the eager per-gate methods (`apply_gate_1q`, `apply_cx`) for a short circuit instead of `run_circuit_jit` |

### Performance

Both `DenseSVSimulator` and `Chunk` need `2**n * 16` bytes just for the statevector --
around 17 GB already at 30 qubits, before anything else. `MPSSimulator` stays at
`O(n * max_bond**2)` instead, so a low-entanglement circuit like the GHZ chain above
runs at 50+ qubits on an ordinary machine where the dense engines would raise
`MemoryError`. Exact wall-clock numbers depend heavily on the machine and circuit
depth, so none are quoted here as a general result -- run `mps.summary()` yourself to
see this machine's own numbers, the same way step 4 above does.

::: dense_evolution.backends.mps

**See also**: [`DenseSVSimulator`](simulator.md) for exact statevector simulation when
entanglement is too high for a bounded bond dimension, and [`Chunk`](chunk.md) for
anti-OOM dense simulation at large qubit counts instead of a bond-dimension truncation.
