# Interop (Qiskit / PennyLane / STIM)

Every bridge on this page goes through OpenQASM 2.0 and this library's own
[`QASMParser`](parser.md), not a bespoke gate-by-gate translator — gate coverage always
matches whatever the parser and simulator already support.

## Step 1. Bring a Qiskit circuit in

```python
import qiskit
from dense_evolution.interop import from_qiskit

qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
circuit = from_qiskit(qc)
circuit.to_tuples()
```

```
[('h', 0), ('cx', 0, 1)]
```

`from_qiskit` exports `qc` to an OpenQASM 2.0 string (`qiskit.qasm2.dumps`) and hands it
straight to `QASMParser` — the same `QASMCircuit` [the parser guide](parser.md) already
covers, from a Qiskit circuit instead of a QASM string you wrote by hand.

## Step 2. Run it in one call

```python
from dense_evolution.interop import run_qiskit_circuit
from qiskit.quantum_info import Statevector

sim, probs = run_qiskit_circuit(qc)
probs.round(4), Statevector(qc).probabilities().round(4)
```

```
(array([0.5, 0. , 0. , 0.5]), array([0.5, 0. , 0. , 0.5]))
```

`run_qiskit_circuit` does Step 1, runs the result on a fresh `DenseSVSimulator`, and
returns `(sim, probabilities)` — already reordered to match Qiskit's own convention, so
it compares directly against `Statevector(qc).probabilities()` with no extra work.

## Step 3. Why that reordering matters

```python
qc2 = qiskit.QuantumCircuit(2)
qc2.x(0)

sim2, probs2 = run_qiskit_circuit(qc2)
probs2.round(4), sim2.get_probabilities().round(4)
```

```
(array([0., 1., 0., 0.]), array([0., 0., 1., 0.]))
```

Same simulator, two different-looking answers for the exact same state: `probs2` (what
Step 2 returns) matches Qiskit — index 1 (`01`, qubit 0 is the *least* significant bit,
Qiskit's convention). `sim2.get_probabilities()` — the simulator's own native output,
with no reordering — puts it at index 2 (`10`, qubit 0 is the *most* significant bit,
[this library's convention everywhere else](simulator.md#details)). `run_qiskit_circuit`
exists specifically to paper over that difference automatically.

## Step 4. PennyLane needs no such reordering

```python
import pennylane as qml
from dense_evolution.interop import run_pennylane_circuit

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def pl_circuit():
    qml.PauliX(wires=0)
    return qml.probs(wires=[0, 1])

sim3, probs3 = run_pennylane_circuit(pl_circuit)
probs3.round(4)
```

```
array([0., 0., 1., 0.])
```

The exact same physical state as Step 3 — `X` on qubit 0 of 2 — lands at index 2 here,
with no bit-reversal applied, and matches `pl_circuit()`'s own PennyLane-native output.
PennyLane's own wire convention already numbers qubit 0 as the most significant bit,
the same as this library — `run_pennylane_circuit` deliberately does **not** reorder,
unlike `run_qiskit_circuit`; the two frameworks genuinely need different treatment.

## Step 5. Run with a real device's own calibrated noise

```python
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from dense_evolution.interop import noise_model_from_qiskit_backend

backend = FakeSherbrooke()
specs = noise_model_from_qiskit_backend(backend)
ecr_specs = [s for s in specs if s['gate'] == 'ecr' and s['qubits'] == [1, 0]]
len(specs), ecr_specs[0]
```

```
(652, {'gate': 'ecr', 'qubits': [1, 0], 'model': 'depolarizing', 'p': 0.007494257741828603})
```

`FakeSherbrooke` carries a real, measured calibration snapshot — `specs` is one entry
per unique (gate, qubit-target) pair with a calibrated error rate, each already shaped
as the `model`/`p`/`qubits` arguments [`NoiseModel.apply_to_sv`](noise.md) expects, so a
simulation can use this device's actual measured error rates instead of a guessed
number.

## Step 6. Cross-check a Clifford circuit against STIM

```python
import dense_evolution as de
from dense_evolution.interop import to_stim

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
stim_circuit = to_stim(circuit.to_tuples(), circuit.n_qubits)
stim_circuit
```

```
stim.Circuit('''
    I 1
    H 0
    CX 0 1
''')
```

`to_stim` maps every Clifford gate (`h`, `x`, `y`, `z`, `s`, `sdg`, `sx`, `id`, `cx`,
`cy`, `cz`) 1:1 onto a native STIM instruction, for cross-validating a circuit against
STIM's own independent stabilizer simulator — useful specifically because STIM is a
different implementation of the same physics, not a wrapper around this library's own
simulator. The leading `I 1` guarantees the STIM circuit has `n_qubits` qubits even when
the highest-indexed one is never otherwise touched.

---

## Details

### These bridges are not differentiable

`from_qiskit`/`from_pennylane` bake every gate parameter into a plain Python `float`
inside the QASM text before parsing — the value leaves any JAX trace it came from.
`jax.grad` through `run_qiskit_circuit`/`run_pennylane_circuit` does not raise: it
silently returns `0.0`, which reads as "already converged" rather than "not wired up".
For a real gradient, call `from_qiskit`/`from_pennylane` for the `QASMCircuit` alone,
before that float-baking happens, and pass it to
[`circuit_to_energy_fn`](autodiff.md) instead.

### What this bridge can't run

Inherited from the OpenQASM 2.0 export itself, not something this layer works around:
no classical control flow (`if`/`while` and mid-circuit-measurement-conditioned gates
are parsed out, never executed); no expansion of composite/custom gates (a Qiskit call
like `mcx` with 3+ controls exports as a named `gate mcx { ... }` definition that parses
cleanly but isn't a primitive this simulator can execute — a call to it is a silent
no-op, same as any unrecognized gate name); and no backend/device registration — this is
a bridge for circuit *execution* you call explicitly, not a Qiskit `BackendV2` or
PennyLane `Device` you point existing framework code at.

### PennyLane's wire-numbering pitfall

By default, PennyLane numbers exported QASM qubits in the order wires are *first
touched* in the circuit, not by their actual wire index — `qml.PauliX(wires=2)` followed
by `qml.CNOT(wires=[2, 1])` exports as `x q[0]; cx q[0],q[1];`, silently renumbering wire
2 to `q[0]` and wire 1 to `q[1]`. `from_pennylane` passes an explicit `wires=` argument
(the device's declared order for a QNode, the tape's own wires sorted ascending for a
bare tape) specifically to prevent this — nothing further to do on the caller's side,
but worth knowing if a circuit's exported QASM ever looks unexpectedly renumbered
outside this bridge.

### Real calibration data, deduplicated per target

`noise_model_from_qiskit_backend` walks `backend.target`, one entry per unique (gate,
qubit-target) pair — never once per occurrence in a circuit, regardless of how many
times that circuit repeats the same gate on the same qubits. This matters: an earlier
version of this bridge (Dense-Evolution-Discovery's Steane-code hardware script) called
`qiskit_aer`'s `NoiseModel.add_quantum_error` once per gate *occurrence* instead, and
`AerSimulator` composes the same Kraus channel with itself on repeated registration for
the same target — on a circuit repeating gates on the same qubits many times, this
produced a combinatorial explosion of Kraus terms that OOM-killed the process. `measure`
is excluded by default (readout error is a post-measurement classical effect, not a
pre-measurement channel); pass a different `skip_gates` to change that.

### On macOS, Qiskit's own object construction can crash the process

Qiskit's `QuantumCircuit.__init__` is a known upstream segfault risk on macOS/arm64
(reproduced on GitHub Actions arm64 runners, Python 3.10-3.12) — `from_qiskit`/
`run_qiskit_circuit`/`noise_model_from_qiskit_backend` all emit a `RuntimeWarning` the
first time they're called on that platform, pointing at the PennyLane bridge as a
workaround (it never constructs a Qiskit object, so it has no exposure to this bug).

::: dense_evolution.interop.qiskit_pennylane

## See Also

- [`QASMParser`](parser.md) — what every bridge on this page actually parses the
  exported QASM text with.
- [Noise](noise.md) — `NoiseModel.apply_to_sv`, what Step 5's `specs` entries are shaped
  to plug into directly.
- [Autodiff](autodiff.md) — `circuit_to_energy_fn`, the real-gradient alternative when
  differentiating through a circuit these bridges brought in.
