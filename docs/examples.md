# Examples

Runnable, end-to-end examples, each lifted from real experiments/tests/docstrings already
in the repository rather than written fresh for this page:

- [Density-matrix ZNE healing](#density-matrix-zne-healing) — from
  [`experiments/matrix_healing_zne.py`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/research/experiments/matrix_healing_zne.py),
  the exact script `zne_density_matrix`'s docstring reports measured numbers from.
- [MPS for low-entanglement circuits](#mps-for-low-entanglement-circuits) — from
  `tests/unit/test_mps.py::test_run_circuit_jit_ghz_chain`.
- [Differentiable VQE](#differentiable-vqe) — from
  `tests/unit/test_autodiff.py::TestCircuitToEnergyFn`.
- [Vector healing](#vector-healing) — from `tests/integration/test_ia_healing.py`, healing a noisy
  vector sequence (not to be confused with density-matrix ZNE healing above).
- [Molecular ground-state energy](#molecular-ground-state-energy) — the from-scratch
  Hartree-Fock engine behind [`dashboard_core.hamiltonians`](api/dashboard_core_hamiltonians.md).
- [Large circuits with Chunk](#large-circuits-with-chunk) — anti-OOM slicing for circuits too
  large for a single dense allocation.
- [Running a QASM circuit through the engine](#running-a-qasm-circuit-through-the-engine) —
  [`dashboard_core.engine`](api/dashboard_core_engine.md).
- [Molecule geometry builders](#molecule-geometry-builders) — more of
  [`dashboard_core.hamiltonians`](api/dashboard_core_hamiltonians.md).
- [QM/MM forces](#qmmm-forces) — [`dashboard_core.qmmm`](api/dashboard_core_qmmm.md).
- [SYK wormhole instance selection](#syk-wormhole-instance-selection) —
  [`dashboard_core.wormhole`](api/dashboard_core_wormhole.md).
- [Gate tuples to QASM](#gate-tuples-to-qasm) —
  [`dashboard_core.qasm_library`](api/dashboard_core_qasm_library.md).
- [Safe qubit limits for this machine](#safe-qubit-limits-for-this-machine) —
  [`dashboard_core.system_limits`](api/dashboard_core_system_limits.md).

Plus one worked example per remaining library module further below (gates, interop, QEC,
entropy, fermions, tight-binding, circuit drawing, measurement, observables, QFT, random
circuits, entangling layers, Trotterization, and differentiable noise).

## Density-matrix ZNE healing

Zero-noise extrapolation on full density matrices: run a circuit at several *scaled* noise
levels, extrapolate back to the zero-noise limit, then project the (generally unphysical)
extrapolated result back onto the physical cone of valid density matrices
([Smolin-Gambetta-Smith 2012](https://arxiv.org/abs/1106.5458)).

The ideal state (`rho_ideal`) is built once here purely to *grade* the result at the end.
It is never fed into the noise ensemble, the extrapolation, or the physical-projection
step — feeding it back in anywhere but the final fidelity check would be oracle access,
not error mitigation.

```python
import numpy as np
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix

N_QUBITS = 2
BASE_P = 0.05
SCALES = (1.0, 2.0, 3.0)
K_TRAJECTORIES = 200


BELL_QASM = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
BELL_CIRCUIT = de.QASMParser().parse(BELL_QASM)


def bell_state_sv():
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit_jit(BELL_CIRCUIT.to_tuples())
    return np.asarray(sim.get_statevector())


def noisy_density_matrix(ideal_sv, p, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, 'depolarizing', p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


ideal_sv = bell_state_sv()
rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

rng = np.random.default_rng(0)
rho_at_scales = jnp.stack([
    noisy_density_matrix(ideal_sv, BASE_P * scale, K_TRAJECTORIES, rng)
    for scale in SCALES
])

raw_fidelity = uhlmann_fidelity(rho_at_scales[0], rho_ideal)          # base-scale, uncorrected
corrected = zne_density_matrix(rho_at_scales, SCALES)                 # Richardson + physical projection
corrected_fidelity = uhlmann_fidelity(corrected, rho_ideal)           # grading only

print(f"raw fidelity:       {raw_fidelity:.4f}")
print(f"corrected fidelity: {corrected_fidelity:.4f}")
```

Measured on this exact script (5 seeds, 400 trajectories each, 2–5 qubits, 5 noise
channels): 96/100 runs improve fidelity, mean delta +0.12. See the
[Changelog](changelog.md) and [`dense_evolution.mitigation`](api/mitigation.md) docstrings
for the full sweep and the honest negative results (predictive-healing coefficient
perturbation was tested and rejected for the matrix case — negligible effect even
amplified 100x).

Run it yourself:

```bash
python experiments/matrix_healing_zne.py
```

## MPS for low-entanglement circuits

`MPSSimulator` keeps a bounded bond dimension (`max_bond`), trading exactness for circuits
that stay low-entanglement (GHZ chains, shallow QAOA layers, most NISQ ansätze) at qubit
counts a dense statevector could never hold. `run_circuit_jit` fuses the whole circuit into
a single `jax.lax.scan`.

```python
import numpy as np
from dense_evolution import MPSSimulator

n = 6
ops = [["h", 0]] + [["cx", q, q + 1] for q in range(n - 1)]  # GHZ chain

mps = MPSSimulator(n_qubits=n, max_bond=8)
mps.run_circuit_jit(ops)

prob = np.abs(np.asarray(mps.contract_to_statevector())) ** 2
print(prob[0], prob[2 ** n - 1])  # both ~0.5, everything else ~0 -- the GHZ signature
```

See [`dense_evolution.mps`](api/mps.md) for the truncated-SVD mechanics and the
`jax.lax.scan` fusion this relies on.

## Differentiable VQE

`circuit_to_energy_fn` turns a parsed circuit into a `(theta, hamiltonian) -> (energy,
statevector)` function that is differentiable end-to-end with `jax.grad`/`jax.value_and_grad`
— no manual parameter-shift rule needed. This runs a short Adam loop against a random
diagonal Hamiltonian.

```python
import numpy as np
import jax
import jax.numpy as jnp
import dense_evolution as de

VQE_QASM = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    'ry(0.5) q[0]; rx(0.5) q[1]; cx q[0],q[1]; rz(0.2) q[1]; cx q[0],q[1]; '
    'ry(0.5) q[0]; rx(0.5) q[1]; measure q -> c;'
)


def random_hamiltonian(n_qubits, seed=7):
    rng = np.random.default_rng(seed)
    values = np.sort(rng.uniform(-2.5, 2.5, 2 ** n_qubits))
    return jnp.diag(jnp.array(values, dtype=jnp.float64))


circ = de.QASMParser().parse(VQE_QASM)
energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
h_matrix = random_hamiltonian(circ.n_qubits)

rng = np.random.default_rng(3)
theta = rng.uniform(-np.pi, np.pi, n_params)

energy_and_grad = jax.jit(jax.value_and_grad(energy_fn, argnums=0, has_aux=True))
m, v = np.zeros(n_params), np.zeros(n_params)
lr, beta1, beta2 = 0.1, 0.9, 0.999

for epoch in range(1, 31):
    (energy, _), grad = energy_and_grad(jnp.asarray(theta), h_matrix)
    grad = np.asarray(grad)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat, v_hat = m / (1 - beta1 ** epoch), v / (1 - beta2 ** epoch)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

print(f"final energy:  {float(energy):.4f}")
print(f"ground state:  {float(jnp.min(jnp.diag(h_matrix))):.4f}")
```

`circuit_to_energy_fn` also accepts an optional `noise=` (`dense_evolution.NoiseSpec`, a
JAX pytree) to trace noisy VQE runs natively — composable with `jax.jit`, `jax.grad`, and
`jax.vmap` over noise-key batches. See [`dense_evolution.autodiff`](api/autodiff.md).

## Vector healing

Not to be confused with [density-matrix ZNE healing](#density-matrix-zne-healing) above —
same word, different thing. This is `dense_evolution.healing`'s Phi-Trigger applied to a
real `(n_steps, dim)` vector sequence (a VQE parameter/energy trajectory, MD telemetry, or
any other noisy sequence): per step, keep it if the change from a local baseline looks like
genuine dynamics, replace it with the local median if it looks like static noise. NaN/Inf
entries are always sanitized first, regardless of that decision.

```python
import numpy as np
from ia_utils.vector_healing import enhanced_dense_healing_hybrid

rng = np.random.default_rng(0)
trajectory = rng.normal(loc=1.0, scale=0.05, size=(30, 4))  # e.g. a VQE parameter history
trajectory[17] = [50.0, -30.0, 12.0, -8.0]                  # one genuine outlier step

healed, meta = enhanced_dense_healing_hybrid(trajectory)

print(f"outlier step replaced: {not np.array_equal(healed[17], trajectory[17])}")
print(f"fallback_triggered:    {meta['fallback_triggered']}")     # NaN/Inf found *and* corrected
print(f"reconstruction_error:  {meta['reconstruction_error']:.4f}")
```

Same primitives, reachable without writing Python: `dashboard_core.run_vector_healing`
(kernel-facing wrapper), the Composer kernel's `POST /api/vector_healing`, or the MCP tool
`dense_evolution_vector_healing` — see [MCP Server](mcp.md).

## Molecular ground-state energy

`build_qubit_hamiltonian` runs a real Hartree-Fock SCF loop from scratch, then hands the
result to PennyLane only for the fermion-to-qubit mapping.

```python
from dense_evolution.native_hf.bridge import build_qubit_hamiltonian

hamiltonian, n_qubits, hf_result = build_qubit_hamiltonian(
    atomic_numbers=[1, 1],
    geometry_angstrom=[[0, 0, 0], [0, 0, 0.7414]],
    n_electrons=2,
)

print(f"qubits: {n_qubits}")
print(f"Hartree-Fock energy: {hf_result.total_energy:.6f} Hartree")
```

```
qubits: 4
Hartree-Fock energy: -1.116684 Hartree
```

`hamiltonian` is the mapped qubit Hamiltonian — feed it into `circuit_to_energy_fn` above for
a full VQE run instead of just Hartree-Fock. See [`dense_evolution.native_hf`](api/native_hf.md).

## Large circuits with Chunk

`Chunk` runs a circuit too large for a single dense allocation by slicing it into RAM-sized
pieces instead of raising `MemoryError`.

```python
from dense_evolution import Chunk

n = 10
circuit = [("h", 0)] + [("cx", q, q + 1) for q in range(n - 1)]

sim = Chunk(n, chunk_size_gates=5)
sim.run_chunk(circuit)
probs = sim.get_probabilities()
print(probs[0], probs[2 ** n - 1])
```

```
0.5 0.5
```

Same GHZ signature as the MPS example above. `chunk_size_gates` controls how many gates run
per RAM-resident slice, not circuit correctness. See [`Chunk`](api/chunk.md) for the
multi-device distributed path (`run_chunk_distributed`).

## Running a QASM circuit through the engine

`run_circuit_from_qasm` parses OpenQASM text and runs it on a real `dense_evolution` engine,
returning every quantity a UI panel or a script would need (probabilities, shot counts,
statevector) in one call.

```python
from dashboard_core.engine import run_circuit_from_qasm

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""
result = run_circuit_from_qasm(qasm, n_shots=1000, seed=0)
print(result.probabilities.round(3).tolist())
print(sorted(set(result.counts.keys())))
```

```
[0.5, 0.0, 0.0, 0.5]
['00', '11']
```

`backend="mps"` runs the same call through `MPSSimulator` instead of `DenseSVSimulator` for
circuits with many low-entanglement qubits. `noise_model`/`noise_p` apply a real
`NoiseModel` Kraus channel and report `fidelity_vs_ideal` against the same circuit run
noiselessly in the same call. See [`dashboard_core.engine`](api/dashboard_core_engine.md).

## Molecule geometry builders

`ring_geometry` places N atoms on a regular polygon with a fixed bond length between
neighbors — the same real geometry used for H3+'s equilateral-triangle ground state.

```python
from dashboard_core.hamiltonians import ring_geometry

print(ring_geometry(3, 0.8738).round(4))
```

```
[[ 0.5045  0.      0.    ]
 [-0.2522  0.4369  0.    ]
 [-0.2522 -0.4369  0.    ]]
```

Feed the result straight into `build_qubit_hamiltonian`'s `geometry_angstrom` argument above
to get a Hamiltonian for a ring molecule instead of a hand-typed geometry. See
[`dashboard_core.hamiltonians`](api/dashboard_core_hamiltonians.md) for `linear_chain_geometry`
and the built-in `MOLECULE_CATALOG`.

## QM/MM forces

`compute_hellmann_feynman_forces` computes the real force on every nucleus of a
`MOLECULE_CATALOG` molecule: F = -d\<psi|H(R)|psi\>/dR, by central finite difference.

```python
from dashboard_core.qmmm import compute_hellmann_feynman_forces
from dashboard_core.hamiltonians import MOLECULE_CATALOG

h2 = [k for k in MOLECULE_CATALOG if k.startswith("H2 ")][0]
result = compute_hellmann_feynman_forces(h2)
print(round(result["energy_hartree"], 4))
print(0.01 < result["force_norm"] < 0.02)
```

```
-1.1373
True
```

The small nonzero `force_norm` at equilibrium is expected, not an error — it is the real
residual of a finite-difference derivative, not exactly zero. Feed forces like these into
`run_md_trajectory` for a full MD loop. See [`dashboard_core.qmmm`](api/dashboard_core_qmmm.md).

## SYK wormhole instance selection

The traversable-wormhole-inspired teleportation protocol needs an SYK Hamiltonian instance
with a specific commuting-pair count (the paper's own selection criterion).
`select_good_instance` screens random seeds and returns the one closest to that target.

```python
import math
from dashboard_core.wormhole import select_good_instance

seed = select_good_instance(n_majorana=8, k_terms=10, J=math.sqrt(2), n_candidates=200, target_commuting=34)
print(seed)
```

```
61
```

That `seed` (with the same `n_majorana`/`k_terms`/`J`) reproduces the same SYK instance for
the full teleportation protocol. See
[`dashboard_core.wormhole`](api/dashboard_core_wormhole.md).

## Gate tuples to QASM

`gate_tuples_to_qasm` goes the other direction from every other example on this page: it
turns a dense_evolution gate-tuple circuit (what `de.qft`/`de.ghz_state`/`de.random_circuit`
return) into real OpenQASM 2.0 text.

```python
from dashboard_core.qasm_library import gate_tuples_to_qasm

print(gate_tuples_to_qasm([("h", 0), ("cx", 0, 1)], n_qubits=2))
```

```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
```

Useful for turning any of dense_evolution's own circuit generators into a QASM preset
without hand-transcribing gates. See
[`dashboard_core.qasm_library`](api/dashboard_core_qasm_library.md).

## Safe qubit limits for this machine

`max_safe_dense_qubits` suggests a maximum qubit count for a dense simulation on the current
machine, from its real available RAM — not a fixed constant.

```python
from dashboard_core.system_limits import max_safe_dense_qubits

limits = max_safe_dense_qubits()
print(sorted(limits.keys()))
print(16 <= limits["max_qubits_dense"] <= 27)
```

```
['available_mb', 'max_qubits_dense', 'threshold_pct', 'total_mb']
True
```

`max_qubits_dense` is floored at 16 and capped at 27 by
`dense_evolution.chunk.get_dynamic_chunk`'s own design — machine-dependent between those
bounds. See [`dashboard_core.system_limits`](api/dashboard_core_system_limits.md).

## Gate matrices

`GATES` is the plain dict of unitary matrices every simulator engine looks up gate names
in — useful directly whenever you need the raw matrix, not just to run a circuit.

```python
import numpy as np
from dense_evolution.gates import GATES

print(np.asarray(GATES["h"]).round(4))
```

```
[[ 0.7071+0.j  0.7071+0.j]
 [ 0.7071+0.j -0.7071+0.j]]
```

See [`dense_evolution.gates`](api/gates.md) for the full static-gate table and
`PARAMETRIC_GATES` for parametric ones (`rx`, `ry`, `rz`, ...).

## Real device noise from a Qiskit backend

`noise_model_from_qiskit_backend` builds a noise specification straight from a Qiskit
`BackendV2`'s own calibration data — a real device's measured per-qubit/per-gate error
rates, not an idealized channel.

```python
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from dense_evolution.interop import noise_model_from_qiskit_backend

backend = FakeSherbrooke()
spec = noise_model_from_qiskit_backend(backend)
print(len(spec))
print(spec[0])
```

```
652
{'gate': 'sx', 'qubits': [0], 'model': 'depolarizing', 'p': 0.00028775142091170115}
```

Each entry is directly usable as the `model`/`p`/`qubits` arguments to
`NoiseModel.apply_to_sv`. See [`dense_evolution.interop`](api/interop.md).

## Erasure-aware QEC decoding

`erasure_aware_decode` corrects errors on qubits KNOWN to have been erased (e.g. a
heralded lost photon) — it can resolve more errors than a standard syndrome-only decoder,
up to *d*-1 on a distance-*d* code instead of floor((*d*-1)/2) (Grassl, Beth & Pellizzari
1997), because it is told where to look instead of having to infer it.

```python
from dense_evolution.qec import compute_syndrome, erasure_aware_decode

x_stabilizers = ["IIIXXXX", "IXXIIXX", "XIXIXIX"]
z_stabilizers = ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]
stabilizers = x_stabilizers + z_stabilizers

error = "IIIZIII"
syndrome = compute_syndrome(error, stabilizers)
result = erasure_aware_decode(syndrome, heralded_qubits=[3], n_qubits=7, stabilizers=stabilizers)
print(result == error)
```

```
True
```

This is the Steane [[7,1,3]] code with the full X+Z stabilizer set — one family alone
cannot distinguish every error type at a heralded qubit, so it returns `None` there rather
than guess. See [`dense_evolution.qec`](api/qec.md) for `pymatching_decode` (standard MWPM,
no erasure info) and `blind_minimum_weight_decode`.

## Entanglement entropy

`von_neumann_entropy` of a qubit's reduced density matrix measures how entangled it is
with the rest of the system — 0 for a product state, ln(2) for one qubit of a Bell pair.

```python
import numpy as np
import dense_evolution as de
from dense_evolution.entropy import partial_trace, von_neumann_entropy

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)

sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = np.asarray(sim.get_statevector())
rho_a = partial_trace(sv, 2, [0])
print(round(von_neumann_entropy(rho_a), 4))
```

```
0.6931
```

`0.6931` is `ln(2)` — qubit 0 of a Bell pair is maximally mixed on its own even though the
joint two-qubit state is pure. See [`dense_evolution.entropy`](api/entropy.md) for
`mutual_information`, which can reveal correlations a single-qubit marginal cannot.

## Majorana-to-Pauli mapping

`majorana_pauli_terms` gives the Jordan-Wigner Pauli term for one Majorana fermion mode —
the building block SYK-model Hamiltonians (like the wormhole protocol above) are built from.

```python
from dense_evolution.fermions import majorana_pauli_terms

print(majorana_pauli_terms(1, 2))
print(majorana_pauli_terms(2, 2))
```

```
(1.0, {0: 'X'})
(1.0, {0: 'Y'})
```

See [`dense_evolution.fermions`](api/fermions.md).

## Zinc-blende tight-binding band structure

`zincblende_hamiltonian` builds the 8x8 sp3 tight-binding Bloch Hamiltonian for a real
two-atom zinc-blende crystal (e.g. GaAs) at a given crystal momentum.

```python
import numpy as np
from dense_evolution.harrison_tb import zincblende_hamiltonian

H = zincblende_hamiltonian((0.0, 0.0, 0.0), "Ga", "As", lattice_constant_angstrom=5.6533)
print(H.shape)
print(np.sort(np.linalg.eigvalsh(H).real).round(3))
```

```
(8, 8)
[-22.069  -9.537  -9.537  -9.537  -6.631  -3.273  -3.273  -3.273]
```

Eigenvalues at Gamma are the band energies (eV) at the center of the Brillouin zone. See
[`dense_evolution.harrison_tb`](api/harrison_tb.md).

## Matplotlib circuit diagrams

`plot_circuit` draws a circuit in dense_evolution's own native tuple format as a
Quirk-style box diagram — the same format every simulator engine's `run_circuit` accepts.

```python
import dense_evolution as de

fig = de.plot_circuit([("h", 0), ("cx", 0, 1)], 2)
fig.savefig("circuit.png", dpi=150, bbox_inches="tight")
```

`fig` is a plain `matplotlib.figure.Figure` — save it, show it, or embed it in a notebook
like any other. See [`dense_evolution.diagram`](api/diagram.md).

## Plain-text circuit diagrams

`draw_circuit` prints an ASCII diagram — no matplotlib needed, safe for a terminal or a
log file regardless of console encoding.

```python
from dense_evolution.drawing import draw_circuit

print(draw_circuit([("h", 0), ("cx", 0, 1)], 2))
```

```
q0: --H----*--
q1: -------X--
```

See [`dense_evolution.drawing`](api/drawing.md).

## Sampling shots from a statevector

`sample_counts` simulates real projective measurement shots from a statevector, returning
a Qiskit-style counts dict — the same thing a real device gives you, not exact
probabilities.

```python
import numpy as np
import dense_evolution as de
from dense_evolution.measurement import sample_counts, statevector_fidelity

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)

sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = np.asarray(sim.get_statevector())

rng = np.random.default_rng(0)
counts = sample_counts(sv, 1000, rng=rng)
print(sorted(counts.keys()))
print(statevector_fidelity(sv, sv))
```

```
['00', '11']
1.0
```

`statevector_fidelity` is the cheap pure-state counterpart to `uhlmann_fidelity` above —
use it whenever both states are pure. See [`dense_evolution.measurement`](api/measurement.md).

## Pauli expectation values

`pauli_expectation` computes `<psi|P|psi>` for a Pauli string `P`, in O(dim) without ever
building the full `2**n_qubits` matrix for `P`.

```python
import numpy as np
import dense_evolution as de
from dense_evolution.observables import pauli_expectation

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)

sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = np.asarray(sim.get_statevector())
print(pauli_expectation(sv, "ZZ"))
```

```
1.0
```

A Bell pair's two qubits are perfectly correlated in Z, even though each one's own `<Z>` is
0. See [`dense_evolution.observables`](api/observables.md) for the dict/pair-list forms of
`pauli_terms` and `pauli_sum_expectation` for a full weighted Hamiltonian at once.

## Quantum Fourier Transform

`qft` returns the gate-tuple circuit for the Quantum Fourier Transform on `n_qubits`.

```python
import numpy as np
import dense_evolution as de

sim = de.DenseSVSimulator(3)
sim.run_circuit_jit(de.qft(3))
print(np.asarray(sim.get_probabilities()).round(4))
```

```
[0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125]
```

QFT on the all-zeros input state gives a uniform superposition over every basis state. See
[`dense_evolution.qft`](api/qft.md).

## Random circuits

`random_circuit` builds a random gate-tuple circuit — useful for benchmarking or fuzzing
any code path that accepts the standard gate-tuple format.

```python
from dense_evolution.circuits.random_circuit import random_circuit

print(random_circuit(3, 5, seed=0))
```

```
[('t', 0), ('cx', 0, 2), ('t', 1), ('tdg', 1), ('z', 2)]
```

Same seed always gives the same circuit. See
[`dense_evolution.random_circuit`](api/random_circuit.md).

## Entangling layers

`entangling_layer` builds the two-qubit gate pattern most NISQ ansätze repeat layer after
layer — `ghz_state` above is built from exactly this, pattern="linear".

```python
from dense_evolution.circuits.topology import entangling_layer

print(entangling_layer(4, pattern="linear", gate="cx"))
```

```
[('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 3)]
```

See [`dense_evolution.topology`](api/topology.md) for other patterns (`"circular"`,
`"all-to-all"`, ...).

## Trotterized Hamiltonian evolution

`trotter_evolve_ops` builds the gate-tuple circuit approximating `exp(-i*H*t)` for
`H = sum_k c_k*P_k`, via the Trotter product formula.

```python
from dense_evolution.trotter import trotter_evolve_ops

ops = trotter_evolve_ops([(1.0, {0: "Z", 1: "Z"})], t=0.5, n_steps=1)
print(ops)
```

```
[('cx', 0, 1), ('rz', 1, 1.0), ('cx', 0, 1)]
```

A ZZ coupling term Trotterizes into a CNOT-RZ-CNOT sandwich — the standard two-qubit
Pauli-rotation gadget. `order=2` gives the more accurate Strang splitting at 2x the gates
per step. See [`dense_evolution.trotter`](api/trotter.md).

## Semiconductor band structure (sp3s*)

`sp3s_star_hamiltonian` builds the 10x10 sp3s* tight-binding Bloch Hamiltonian for any
material in `MATERIALS` (real published parameters), and `direct_gap_at_gamma` reads off
the Gamma-point band gap directly from it.

```python
from dense_evolution.solvers.vhd_tb import direct_gap_at_gamma, MATERIALS

print(sorted(MATERIALS.keys()))
print(round(direct_gap_at_gamma("Si"), 4))
```

```
['AlAs', 'AlP', 'AlSb', 'C', 'GaAs', 'GaP', 'GaSb', 'Ge', 'InAs', 'InP', 'InSb', 'Si', 'SiC', 'Sn', 'ZnSe', 'ZnTe']
3.43
```

Silicon is indirect-gap, so this Gamma-Gamma value is not its true (lower, off-Gamma)
fundamental gap — see the function's own docstring for which materials this value is
directly meaningful for. See [`dense_evolution.vhd_tb`](api/vhd_tb.md).

## Adversarial robustness test for vector healing

`craft_adversarial_healing_perturbation` crafts the minimal perturbation, within an L2
epsilon-ball, that flips `enhanced_dense_healing_hybrid`'s Phi-Trigger decision at one
point — a targeted robustness stress test, not random noise.

```python
import numpy as np
from ia_utils.adversarial_vector_attack import craft_adversarial_healing_perturbation

rng = np.random.default_rng(0)
vectors = rng.normal(1.0, 0.05, size=(30, 3))
result = craft_adversarial_healing_perturbation(vectors, target_idx=15, epsilon=0.1)
print(sorted(result.keys()))
```

```
['final_magnitude', 'final_trigger_active', 'original_magnitude', 'original_trigger_active', 'perturbation_norm', 'perturbed_vettori', 'success']
```

See [`ia_utils.adversarial_vector_attack`](api/ia_utils_adversarial_vector_attack.md).

## Differentiable noise as a JAX pytree

`NoiseSpec` wraps a noise configuration as a real JAX pytree, so noise parameters thread
through `jax.jit`/`jax.grad`/`jax.vmap` natively — passed as `circuit_to_energy_fn`'s
`noise=` argument above, instead of being applied as an external Python-side step.

```python
import jax
from dense_evolution.registry import NoiseSpec

key = jax.random.PRNGKey(0)
spec = NoiseSpec(model="depolarizing", p=0.05, jax_key=key, qubits=[0, 1])
print(spec)
```

```
NoiseSpec(model='depolarizing', p=0.05, qubits=(0, 1))
```

`model`/`qubits` are static; `p`/`jax_key` are real pytree leaves — `p` itself can be a
traced/differentiable value. See [`dense_evolution.registry`](api/registry.md).
