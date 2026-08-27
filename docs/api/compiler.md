# Compiler

`QuantumTranspiler` rewrites a circuit's non-native gates into ones the simulator's fast
path actually knows — `run_circuit` calls it automatically (see [Gates](gates.md)'s
Details for which gate names have no `GATE_IDS` entry at all: `swap`, `ccx`, `ecr`,
`iswap`, `u2`, `u3`, exactly the gates this page decomposes).

## Step 1. Transpile a circuit by hand

```python
import dense_evolution as de
from dense_evolution.circuits.compiler import QuantumTranspiler

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; h q[0]; swap q[0],q[1]; ccx q[0],q[1],q[2];'
circuit = de.QASMParser().parse(qasm)
expanded = QuantumTranspiler.transpile(circuit.to_tuples())
len(expanded), expanded[:4]
```

```
(19, [('h', 0), ('cx', 0, 1), ('cx', 1, 0), ('cx', 0, 1)])
```

3 gates went in, 19 came out: `h` passed through unchanged, `swap` became 3 `cx` gates,
`ccx` became 15 more. `run_circuit` runs exactly this step first, automatically, unless
called with `transpile=False` — every step below is one piece of what just happened.

## Step 2. Toffoli (CCX) → 15 native gates

```python
QuantumTranspiler.decompose_toffoli(0, 1, 2)
```

```
[('h', 2), ('cx', 1, 2), ('tdg', 2), ('cx', 0, 2), ('t', 2), ('cx', 1, 2), ('tdg', 2),
 ('cx', 0, 2), ('t', 1), ('t', 2), ('cx', 0, 1), ('h', 2), ('t', 0), ('tdg', 1), ('cx', 0, 1)]
```

The standard Barenco T/Tdg/CX construction: 6 `cx` plus 7 single-qubit gates (`h`, `t`,
`tdg`), 15 total, for one 3-qubit Toffoli — the two controls are `c1`, `c2`, the target
is `t`.

## Step 3. SWAP → 3 CX

```python
QuantumTranspiler.decompose_swap(0, 1)
```

```
[('cx', 0, 1), ('cx', 1, 0), ('cx', 0, 1)]
```

The textbook identity `SWAP = CX(a,b)·CX(b,a)·CX(a,b)` — no ancilla, no extra qubits,
just three CX gates in alternating direction.

## Step 4. iSWAP → {S, H, CX}, verified exact

```python
import numpy as np

qasm2 = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; rx(0.7) q[1]; cx q[0],q[1];'
circuit2 = de.QASMParser().parse(qasm2)

sim_direct = de.DenseSVSimulator(2)
sim_direct.run_circuit(circuit2.to_tuples())
sim_direct.apply_gate_2q(de.GATES['iswap'], 0, 1)

sim_decomp = de.DenseSVSimulator(2)
sim_decomp.run_circuit(circuit2.to_tuples())
sim_decomp.run_circuit(QuantumTranspiler.decompose_iswap(0, 1), transpile=False)

float(np.max(np.abs(sim_direct.get_statevector() - sim_decomp.get_statevector())))
```

```
1.1102230246251565e-16
```

`decompose_iswap(q1, q2)` returns
`[('s', q1), ('h', q1), ('s', q2), ('cx', q1, q2), ('cx', q2, q1), ('h', q2)]` — applying
those 6 gates gives the exact same statevector as applying `GATES['iswap']` directly,
down to floating-point noise, not merely a physically-equivalent state up to global
phase.

## Step 5. ECR → {S, SX, X, CX, GPhase}

```python
QuantumTranspiler.decompose_ecr(0, 1)
```

```
[('s', 0), ('sx', 1), ('cx', 0, 1), ('x', 0), ('gphase', 0, -0.7853981633974483)]
```

The `{s, sx, x, cx}` part alone reproduces ECR only up to a constant `e^{i pi/4}` global
phase; the trailing `gphase` gate (`-pi/4` here) supplies exactly the correction needed
to match `GATES['ecr']` exactly, verified the same way Step 4 verifies `iswap` — direct
gate versus decomposed circuit, same statevector to machine precision.

## Step 6. U3/U2 → {Rz, Ry, Rz, GPhase}

```python
QuantumTranspiler.decompose_u3(0, 0.3, 0.5, 0.7)
```

```
[('rz', 0, 0.7), ('ry', 0, 0.3), ('rz', 0, 0.5), ('gphase', 0, 0.6)]
```

`U3(theta, phi, lam) = e^{i(phi+lam)/2} * Rz(phi)*Ry(theta)*Rz(lam)` — a standard ZYZ
decomposition, applied rightmost-first (`rz(lam)` then `ry(theta)` then `rz(phi)`), with
`gphase` supplying the leading `e^{i(phi+lam)/2}` scalar so the result matches
`PARAMETRIC_GATES['u3'](theta, phi, lam)` exactly rather than up to an unobservable
phase. `decompose_u2(q, phi, lam)` is the same function called with `theta` fixed to
`pi/2` — `U2(phi, lam) = U3(pi/2, phi, lam)`.

---

## Details

### What `gphase` actually does

`gphase(q, alpha)` multiplies the *entire* n-qubit statevector by the scalar `e^{i alpha}`,
not just qubit `q`'s local `|1>` amplitude the way `p`/`u1` (phase, `GATE_IDS` entry 12)
does — applying `e^{i alpha} * I` inside any single qubit's 2x2 subspace is mathematically
identical to scaling the whole state by `e^{i alpha}`, since it commutes with every other
qubit's identity. It only ever appears as output from `decompose_u3`/`decompose_u2` —
nothing in this library emits it any other way.

### `transpile` leaves everything else untouched

Only `ccx`, `swap`, `iswap`, `ecr`, and parametric `u2`/`u3` get expanded; every other
gate name in the input list is passed straight through unchanged, in place, in order.

### A structural-only transpile pass (no real angles yet)

Some callers — `dense_evolution.solvers.autodiff`'s VQE template builder — run `transpile`
on tuples that carry qubit indices but no parameter values yet (angles get substituted
in later, from a traced `theta` during optimization). `decompose_u2`/`decompose_u3` can't
run without real `theta`/`phi`/`lam` values (needed to compute the `gphase` angle), so a
`u2`/`u3` tuple that's too short to actually contain them is passed through unchanged
instead of raising a confusing `TypeError` here — a caller that genuinely can't support
`u2`/`u3` templates raises its own clearer error further down its own pipeline.

### The actual compiled kernel this all feeds

`_apply_gate_fast_step` (one gate) and `_run_circuit_scan_core` (a whole circuit via
`jax.lax.scan`) are this module's private JAX-jitted engine — what
[`DenseSVSimulator.run_circuit_jit`](simulator.md) actually calls, dispatching on the
same numeric gate IDs [Gates](gates.md) documents (`GATE_IDS`) rather than string names,
via `jax.lax.switch`/`cond` so the whole circuit compiles to one XLA call. Two public-ish
wrappers exist: `_compile_and_run_circuit_jit` (safe to call repeatedly with the same
input buffer) and `_compile_and_run_circuit_jit_donated` (marks its statevector argument
donated — faster, but only safe where the caller immediately rebinds its own reference
and never reads the old buffer again, which is exactly `run_circuit_jit`'s
`self.sv = ...` pattern).

### Bug history worth knowing if you're reading this kernel's source

The 1-qubit and 2-qubit fast-path kernels originally indexed qubits LSB-first
(`1 << q`) while every other part of the simulator ([MSB-first](simulator.md#details)) —
silently wrong on any multi-qubit circuit not symmetric under qubit reversal, fixed by
computing the same `n_qubits - 1 - qubit` physical position everywhere else does. `CP`
and `CRZ` (gate IDs 22 and 25) used to be silently merged under one ID even though
they're different gates (`CP` phases only the `|11>` component; `CRZ` phases the target
conditioned on its own bit) — `CRZ` wasn't reachable at all before this was split apart.

::: dense_evolution.circuits.compiler

## See Also

- [`DenseSVSimulator`](simulator.md) — `run_circuit`/`run_circuit_jit`, the methods that
  call `transpile` and the compiled kernel automatically.
- [Gates](gates.md) — `GATE_IDS`'s exact coverage, and the `GATES`/`PARAMETRIC_GATES`
  matrices every decomposition here is verified against.
- [`QASMParser`](parser.md) — produces the tuple lists `transpile` and every method on
  this page operate on.
