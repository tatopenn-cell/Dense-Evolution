```
██████╗ ███████╗███╗   ██╗███████╗███████╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝
██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗  
██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
██████╔╝███████╗██║ ╚████║███████║███████╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
███████╗██╗   ██╗ ██████╗ ██╗     ██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║   ██║██╔═══██╗██║     ██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
█████╗  ██║   ██║██║   ██║██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝██║   ██║██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
███████╗ ╚████╔╝ ╚██████╔╝███████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝  ╚═══╝   ╚═════╝ ╚══════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```


**Dense Statevector Quantum Simulator · JAX XLA · NISQ · VQE · QML**

[![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square)](https://tatopenn-cell.github.io/Dense-Evolution/)
[![codecov](https://codecov.io/gh/tatopenn-cell/Dense-Evolution/branch/main/graph/badge.svg)](https://codecov.io/gh/tatopenn-cell/Dense-Evolution)
[![PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSL_1.1-orange?style=flat-square)](LICENSE.md)
[![Build](https://img.shields.io/badge/Build-Passing-00ff9d?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/actions)
[![Cross-Validation CI](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/tatopenn-cell/Dense-Evolution?style=flat-square&color=blueviolet)](https://github.com/tatopenn-cell/Dense-Evolution/releases)
[![Last Commit](https://img.shields.io/github/last-commit/tatopenn-cell/Dense-Evolution?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/commits/main)
[![Issues](https://img.shields.io/github/issues/tatopenn-cell/Dense-Evolution?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/issues)
[![Stars](https://img.shields.io/github/stars/tatopenn-cell/Dense-Evolution?style=flat-square&color=yellow)](https://github.com/tatopenn-cell/Dense-Evolution/stargazers)
[![JAX](https://img.shields.io/badge/Backend-JAX_XLA-f9ab00?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)

---

## ▍ What It Is

**Dense Evolution** is a high-performance statevector simulator engineered for deep NISQ circuits, VQE pipelines, and QML workloads. It eliminates Kronecker product overhead entirely via stride-sliced linear kernel fusion compiled through JAX XLA — keeping memory at the theoretical minimum of `2ⁿ × 16 bytes`.

📖 **[Full documentation, API reference, and worked examples →](https://tatopenn-cell.github.io/Dense-Evolution/)**

A Streamlit dashboard (`app_dashboard.py`) provides live telemetry across 8 panels per simulation run — Quantum Simulator and Vector Healing tabs, run locally with `streamlit run app_dashboard.py`. `legacy/dash.py` is the original Google Colab notebook this was ported from, kept for reference only (not installable — see the file header).

---

## ▍ Install

```bash
pip install dense-evolution  # JAX is a core dependency, installed by default

# full stack: GPU · dashboard · Qiskit/PennyLane interop
pip install dense-evolution[full]

# just the interop bridge
pip install dense-evolution[qiskit]
pip install dense-evolution[pennylane]

# Composer's local kernel (see "Composer" below)
pip install dense-evolution[composer]

# MCP server for the Composer kernel (see "MCP Server" below)
pip install dense-evolution[mcp]

# development
git clone https://github.com/tatopenn-cell/Dense-Evolution.git
cd Dense-Evolution && pip install -e .[full]
```

**Google Colab (3 lines):**

```python
!git clone https://github.com/tatopenn-cell/Dense-Evolution.git
%cd Dense-Evolution
!pip install -e .
```

---

## ▍ Quick Start

```python
from dense_evolution import DenseSVSimulator, QASMParser

# parse any OpenQASM 2.0 / 3.0 string
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"""

parser = QASMParser()
circuit = parser.parse(qasm)

sim = DenseSVSimulator(n_qubits=3)
sim.run_circuit_jit(circuit.to_tuples())

probs = sim.get_probabilities()
sv    = sim.get_statevector()
```

**Dashboard (local, Streamlit):**

```bash
pip install "dense-evolution[dashboard]"  # JAX already included by default
streamlit run app_dashboard.py
```

**Anti-OOM for large circuits:**

```python
from dense_evolution import Chunk

sim = Chunk(27)                                    # logical 27 qubits
circuit_ops = [['h', i] for i in range(27)]
sim.run_chunk(circuit_ops, chunk_size_gates=500)   # SafeMemoryGuard active
```

---

## ▍ Architecture

```
dense_evolution/
├── registry.py       hardware detection · JAX/CuPy/NumPy flags · NoiseModel (Kraus channels)
├── gates.py          GATES{} · PARAMETRIC_GATES{} · GATE_IDS{}
├── healing.py        predictive state engine · Phi_AB · vettore dinamico · MemoryReflectionEngine
├── parser.py         QASMParser · QASMCircuit · OpenQASM 2.0 / 3.0
├── compiler.py       QuantumTranspiler · _apply_gate_fast_step (jit) · gate decomposition
├── chunk.py          SafeMemoryGuard · MemoryChunker · CircuitChunker · Chunk (Anti-OOM)
├── simulator.py      DenseSVSimulator · run_batch_jit · vmap batch VQE
├── mps.py            MPSSimulator — matrix-product-state backend, JAX-backed
├── autodiff.py        circuit_to_energy_fn — the real, public VQE gradient engine
├── mitigation.py      Zero-Noise Extrapolation — richardson_extrapolate, zero_noise_extrapolation, zne_density_matrix (+ jit variants)
├── interop.py         run_qiskit_circuit / run_pennylane_circuit / from_qiskit / from_pennylane
├── observables.py     Pauli-string expectation values, O(2ⁿ) direct from a statevector
├── measurement.py     statevector → finite-shot counts, sampling helpers
├── states.py          common state-preparation circuits (Bell, GHZ, W, ...) as gate tuples
├── topology.py        entangling_layer — linear/circular/full/star/brick patterns
├── qft.py             Quantum Fourier Transform circuit builder
├── fermions.py        majorana_pauli_terms — Majorana-fermion → qubit (Jordan-Wigner) mapping
├── entropy.py         partial_trace · von_neumann_entropy · mutual_information (multi-qubit, MSB-first)
├── trotter.py         pauli_rotation_ops · trotter_evolve_ops — real-time Hamiltonian evolution as gates
├── random_circuit.py  random circuit generation for benchmarking/fuzz-testing
├── drawing.py         plain-text circuit diagrams (ASCII, console-safe)
├── harrison_tb.py     sp3 tight-binding Hamiltonians, Harrison universal parameter table
├── vhd_tb.py          sp3s* tight-binding, Vogl-Hjalmarson-Dow material-specific parameters
└── cli.py             `dense-evolution` console script — serve · offline-composer · mcp

ia_utils/
└── vector_healing.py   median_healing · enhanced_dense_healing_hybrid (NaN/Inf-safe, lazy JAX import)

dashboard_core/ + app_dashboard.py + ui_pages/     Streamlit dashboard — VQE engine · QM/MM · MD simulation · 3D wavefunction
dashboard_core/wormhole.py                          binary sparse SYK model — traversable-wormhole-inspired teleportation (see "MCP Server" below)
local_site/app/server.py                           Composer's local FastAPI compute kernel (see "Composer" below)
mcp_server/server.py                                dense_evolution_mcp — MCP adapter over the Composer kernel (see "MCP Server" below)
research/wormhole_syk.py                            the original wormhole research reproduction + its own verification suite, reference only (not installed as a module — see research/wormhole_syk.md)
legacy/dash.py                                      original Colab notebook, reference only (not installed as a module)
```

**Data flow per run:**

```
▶ Run
└─ core_calcolo_quantistico()        parse → JIT execute → apply noise
    ├─ ottimizza_vqe()               Hellmann-Feynman AD → ADAM → df_vqe_telemetry
    ├─ run_md_simulation_dummy()     QM/MM dynamics → df_md_telemetry + Pearson matrix
    └─ build_panel_*(res)            matplotlib figure → display()
```

---

## ▍ Core Features

| Feature | Detail |
|---|---|
| **Linear Kernel Fusion** | Stride-sliced tensor ops via JAX XLA — zero Kronecker matrices |
| **Parametric Batch JIT** | `run_batch_jit()` evaluates full parameter grids in one `jax.vmap` + `jax.jit` call |
| **Circuit Chunking** | Fixed-size JIT blocks eliminate tracer overhead on 1000+ gate circuits |
| **Kraus Noise Channels** | `depolarizing` `amplitude_damping` `phase_damping` `bitflip` `combined` — stochastic, O(2ⁿ) cost |
| **VQE + ADAM** | Hellmann-Feynman gradient · positional parameter injection into any OpenQASM 2.0 circuit |
| **Anti-OOM Engine** | `SafeMemoryGuard` blocks execution before JAX raises `RESOURCE_EXHAUSTED` |
| **Predictive Healing** | `healing.py` — Φ_AB alignment, dynamic vector, Σ-sync, `MemoryReflectionEngine` |
| **Vector Sequence Healing** | `ia_utils/` — `median_healing`, `enhanced_dense_healing_hybrid` — NaN/Inf-safe, lazy JAX import |
| **Backend Agnostic** | NumPy CPU · JAX XLA CPU/TPU · CuPy CUDA — runtime selection, zero code changes |
| **Live Dashboard** | 8-panel ipywidgets telemetry: probability, VQE energy, entropy, purity, gradient, noise, θ-correction, Pearson heatmap |

---

## ▍ Scientific Validation & Applications

To demonstrate the numerical accuracy and stability of **Dense Evolution**, the simulator was stress-tested across 3,500 continuous spatial sampling points to compute the **Silicon Dimer (Si2) Dissociation Curve** via Variational Quantum Eigensolver (VQE).

* Physical Accuracy: The simulation successfully maps the exact Born-Oppenheimer Potential Energy Curve (PEC), capturing the deep quantum ground state bound minimum at ~3.55 Å with negative total energy, before converging asymptotically toward full molecular dissociation.
* Numerical Precision: Calculations are locked at Double Precision (float64), proving the simulator's resilience against cumulative machine epsilon errors (~ 1.11 × 10⁻¹⁶) across thousands of sequential circuit executions.
* Run this molecular experiment instantly on Google Colab Free Tier:
  [Open Notebook on Google Colab](https://colab.research.google.com/drive/1cX7vYsVaxO29677ltgDTbh3pqUi0NYC5#scrollTo=Qg_lqX-Iw_UM)

---

```text
============================================================
🔬 MOLECULAR VQE: EXACT POTENTIAL ENERGY CURVE (PEC)
============================================================
Distanza R: 1.200 Å | Energia Totale Molecola: +155.761158 eV
Distanza R: 1.671 Å | Energia Totale Molecola: +34.372692 eV
Distanza R: 2.142 Å | Energia Totale Molecola: +6.583098 eV
Distanza R: 2.614 Å | Energia Totale Molecola: +0.727422 eV
Distanza R: 3.085 Å | Energia Totale Molecola: -0.253226 eV
Distanza R: 3.557 Å | Energia Totale Molecola: -0.273498 eV
Distanza R: 4.028 Å | Energia Totale Molecola: -0.170948 eV
Distanza R: 4.500 Å | Energia Totale Molecola: -0.093048 eV
```

#### Variational Quantum Chemistry Plot
Below is the physical validation plot showing the Born-Oppenheimer potential energy curve:

<img width="993" height="593" alt="image" src="https://github.com/user-attachments/assets/5fe57865-40f2-4930-9e8d-63959ea93a22" />


👉 *For the full suite of physical benchmarks, including the Transverse Field Ising Model (TFIM) and Phase Transition mappings, visit the main [Dense-Evolution-Ising-Tests](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests) repository. You can also view the raw script for this specific molecular run [here](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/blob/main/vqe_silicon_molecular.py).*

---

## ▍ API Reference

### `DenseSVSimulator`

```python
sim = DenseSVSimulator(
    n_qubits   : int,
    use_gpu    : bool = False,
    use_float32: bool = False,
)
```

| Method | Description |
|---|---|
| `set_initial_state(state=None)` | Reset to `\|0⟩ⁿ` or inject custom statevector |
| `run_circuit(circuit, transpile=True)` | Plain (non-JIT) gate execution — takes the tuple format below |
| `run_circuit_jit(circuit)` | JIT-compiled gate execution — primary execution path (renamed from `run_circuit_jit_beast_mode` in 8.1.46; old name still works, deprecated) |
| `run_circuit_with_chunking(circuit, chunk_size=500)` | Chunked execution for long circuits |
| `run_batch_jit(base_circuit, parameter_batch)` | `vmap` over parameter grid — returns full batch of statevectors (renamed from `run_parametric_batch_jit` in 8.1.46; old name still works, deprecated) |
| `get_probabilities()` → `np.ndarray` | `\|ψ_i\|²` for all basis states |
| `get_statevector()` → `np.ndarray` | Full complex statevector |
| `measure(qubit_idx)` → `int` | Projective measurement with state collapse |
| `memory_mb()` → `float` | Current RAM usage in MB |
| `apply_gate_1q(gate, qubit)` | Apply arbitrary 2×2 unitary |
| `apply_gate_2q(gate, q1, q2)` | Apply arbitrary 4×4 unitary |

### `QASMParser`

```python
parser  = QASMParser()
circuit = parser.parse(qasm_str)   # → QASMCircuit
valid, msg = parser.validate(circuit)
```

`QASMCircuit` fields: `n_qubits`, `n_cbits`, `ops` (list of gate dicts, e.g.
`{'name': 'h', 'qubits': [0], 'params': []}`). Use `circuit.to_tuples()` to
convert `ops` to the `(name, qubit0[, qubit1, ...][, param0, ...])` tuple
format that `run_circuit` / `run_circuit_jit` expect — don't build that
format by hand.

### `NoiseModel`

```python
noise = NoiseModel()
noise.apply_to_sv(sv, n=4, model='depolarizing', p=0.01, rng=rng)
desc  = NoiseModel.kraus_description('amplitude_damping')
```

### End-to-end: parse → run → apply noise

```python
parser  = QASMParser()
circuit = parser.parse(qasm_str)                  # → QASMCircuit
sim     = DenseSVSimulator(n_qubits=circuit.n_qubits)
sim.run_circuit(circuit.to_tuples())               # dicts -> tuples, then execute

sv_noisy = NoiseModel().apply_to_sv(
    sim.get_statevector(), n=circuit.n_qubits, model='depolarizing', p=0.01,
    rng=np.random.default_rng(42),
)
```

### `Chunk` (Anti-OOM)

```python
sim = Chunk(
    n_qubits         : int,
    chunk_size_gates : int   = 500,
    memory_threshold : float = 0.15,   # block below 15% free RAM
    use_gpu          : bool  = False,
    use_float32      : bool  = False,
)
sim.run_chunk(circuit, chunk_size_gates=500)
```

Backward-compatibility aliases: `chunk1 = MemoryChunker`, `chunk2 = Chunk`, `Chunk2Incrociato = Chunk`.

### `ia_utils.vector_healing`

```python
from ia_utils.vector_healing import median_healing, enhanced_dense_healing_hybrid

healed, radius   = median_healing(vettori, radius_baseline=None)
healed, metadata = enhanced_dense_healing_hybrid(vettori, radius_baseline=None)
```

See **IA Utils — Vector Sequence Healing** above for details.

---

## ▍ Gate Library

**Fixed gates** (no parameters):

| Gate | Symbol | Gate | Symbol |
|---|---|---|---|
| `h` | Hadamard | `x` | Pauli-X |
| `y` | Pauli-Y | `z` | Pauli-Z |
| `s` | S gate | `sdg` | S† gate |
| `t` | T gate | `tdg` | T† gate |
| `sx` | √X gate | `id` | Identity |
| `cx` | CNOT | `cz` | CZ |
| `cy` | CY | `swap` | SWAP |
| `iswap` | iSWAP | `ecr` | ECR |
| `ccx` | Toffoli | | |

**Parametric gates**:

| Gate | Parameters | Description |
|---|---|---|
| `rx(θ)` | θ | X-rotation |
| `ry(θ)` | θ | Y-rotation |
| `rz(θ)` | θ | Z-rotation |
| `p(λ)` | λ | Phase gate |
| `u1(λ)` | λ | U1 (≡ p) |
| `u2(φ, λ)` | φ, λ | U2 rotation |
| `u3(θ, φ, λ)` | θ, φ, λ | Generic single-qubit |
| `cp(λ, ctrl, tgt)` | λ | Controlled-Phase |
| `crz(λ, ctrl, tgt)` | λ | Controlled-RZ |

---

## ▍ Interop — Qiskit / PennyLane

Run a circuit you already wrote in Qiskit or PennyLane on Dense-Evolution's simulator, no manual gate-by-gate rewrite. Both bridges go through OpenQASM 2.0 (`qiskit.qasm2.dumps` / `qml.to_openqasm`) and the existing `QASMParser` — not a bespoke translator, so gate coverage matches whatever the parser/simulator already support (see Gate Library above).

```python
from qiskit import QuantumCircuit
from dense_evolution import run_qiskit_circuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

sim, probs = run_qiskit_circuit(qc)   # probs already in Qiskit's own bit order
```

```python
import pennylane as qml
from dense_evolution import run_pennylane_circuit

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

sim, probs = run_pennylane_circuit(circuit)   # no reordering needed, see below
```

`from_qiskit(circuit)` / `from_pennylane(circuit)` return a `QASMCircuit` (structural conversion only) for anyone who wants to manage their own `DenseSVSimulator`/`Chunk` instead of the convenience runners above.

**Bit-order — read this before comparing arrays across frameworks.** Qiskit indexes probability/statevector arrays little-endian (qubit 0 = least-significant bit); Dense-Evolution indexes MSB-first everywhere (`phys = n_qubits - 1 - qubit`, the same convention `apply_gate_1q`/`apply_gate_2q`/`measure`/beast-mode use). `run_qiskit_circuit` reorders its output into Qiskit's own convention so it's directly comparable to `Statevector(circuit).probabilities()`. PennyLane's own wire convention already matches Dense-Evolution's MSB-first indexing natively — `run_pennylane_circuit` does **not** reorder, on purpose; verified directly on an asymmetric circuit that the two frameworks genuinely need different treatment here, not just "symmetric for simplicity."

**Known limits** (inherited from the QASM2 bridge, not something this layer works around):
- No classical control flow — `if`/`while` and mid-circuit-measurement-conditioned gates are parsed out, not executed (same limitation as native QASM3 circuits, see Changelog v8.1.13).
- No expansion of composite/custom gates. A Qiskit call like `mcx` with 3+ controls gets exported as a named `gate mcx { ... }` definition; the definition is parsed cleanly (no longer corrupts what follows it) but the gate itself isn't a primitive Dense-Evolution knows how to execute, so a call to it is a silent no-op — same as referencing any unrecognized gate name elsewhere in this simulator. Stick to the gates in the Gate Library table above for results you can trust.
- Only a plugin/backend-free bridge — no `qiskit.providers.BackendV2` or PennyLane `Device` registration, so you still call `run_qiskit_circuit`/`run_pennylane_circuit` explicitly rather than pointing existing framework code at a new backend/device string.
- **`run_qiskit_circuit`/`run_pennylane_circuit` themselves are not differentiable.** `from_pennylane`/`from_qiskit` materialize every gate parameter into a plain Python `float` inside the QASM text before parsing — the value leaves the JAX trace entirely. `jax.grad` through `run_pennylane_circuit` does **not** raise: it silently returns `0.0`, which looks like "already converged" rather than "not wired up." Verified directly, not just documented from a guess. **For a real gradient, use `circuit_to_energy_fn` instead** (see next section) — pass it the `QASMCircuit` that `from_qiskit`/`from_pennylane` returns, before that float-baking happens, and `jax.grad` works correctly (also verified directly, on a PennyLane circuit imported this exact way).

---

## ▍ Differentiable Circuits — `circuit_to_energy_fn`

The real VQE gradient engine (`jax.value_and_grad` through a `jax.lax.scan`-based circuit template, verified against finite differences to ~1e-11) used to live only inside `dashboard_core.py`, unreachable from outside the Streamlit app. It's now public, dependency-light (needs only the base `dense-evolution` install, no dashboard/pandas/streamlit), and composes directly with the interop bridge above:

```python
import jax
from dense_evolution import QASMParser, circuit_to_energy_fn

circ = QASMParser().parse("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; rx(0.5) q[0];")
energy_fn, n_params = circuit_to_energy_fn(circ, circ.n_qubits)

h_matrix = ...  # your Hamiltonian, shape (2**n_qubits, 2**n_qubits)
theta = jax.numpy.zeros(n_params)

(energy, statevector), grad = jax.value_and_grad(energy_fn, argnums=0, has_aux=True)(theta, h_matrix)
```

`energy_fn(theta, h_matrix, stato_zero=None) -> (energy, statevector)` is a pure JAX function differentiable in `theta`; `stato_zero` defaults to `|0...0⟩`. `circuit` can come from `QASMParser.parse`, `from_qiskit`, or `from_pennylane` interchangeably — this is what closes the interop bridge's non-differentiability gap noted above.

---

## ▍ Noise Models

All channels applied as post-circuit stochastic Kraus operations on the full statevector.

| Model | Kraus operators | Physical process |
|---|---|---|
| `ideal` | `I` | Noiseless |
| `depolarizing` | `{√(1−p)I, √(p/3)X, √(p/3)Y, √(p/3)Z}` | Isotropic Pauli error |
| `amplitude_damping` | `{K₀=diag(1,√(1−γ)), K₁=[[0,√γ],[0,0]]}` | T₁ energy relaxation |
| `phase_damping` | `{K₀, K₁}` | T₂ dephasing |
| `bitflip` | `{√(1−p)I, √p·X}` | Bit flip σₓ |
| `combined` | depolarizing(p/2) ∘ amplitude_damping(p/3) | Worst-case NISQ |

Fidelity metrics computed on every noisy run: Bhattacharyya `F = Σᵢ √(pᵢqᵢ)` and TVD `= ½Σᵢ|pᵢ−qᵢ|`.

**Native integration with `circuit_to_energy_fn` via `NoiseSpec`.** Applying noise used to mean stepping outside the traced circuit computation: build the ideal statevector, exit the trace, call `apply_to_sv` separately, manage a PRNG key by hand around the training loop. `NoiseSpec` is a real JAX PyTree (`model`/`qubits` static, `p`/`jax_key` as leaves) that `circuit_to_energy_fn`'s `energy_fn` accepts directly as a fourth argument — noise is applied natively, in the same traced graph as `theta`, fully `jax.jit`/`jax.grad`/`jax.vmap`-composable:

```python
import jax
from dense_evolution import QASMParser, circuit_to_energy_fn, NoiseSpec

energy_fn, n_params = circuit_to_energy_fn(circuit, n_qubits)
noise = NoiseSpec(model='depolarizing', p=0.05, jax_key=jax.random.PRNGKey(0))

energy, sv = energy_fn(theta, h_matrix, noise=noise)          # applied inline
energy, sv = jax.jit(energy_fn)(theta, h_matrix, None, noise) # jit-compatible
grad = jax.grad(lambda t: energy_fn(t, h_matrix, None, noise)[0])(theta)  # grad flows through the noise
```

A fresh, independent, reproducible noise draw per call is a `jax.random.split` away — no OS-entropy fallback, no external key bookkeeping. `p` is also a tracked leaf (you *can* differentiate through it), though the gradient is ~0 almost everywhere in practice: the underlying channels sample via a hard threshold (`fire = r < p`), which isn't usefully differentiable without a smooth relaxation — out of scope here.

**`apply_to_sv`'s randomness arguments** — `rng: np.random.Generator` and `jax_key: jax.random.PRNGKey` — cover the two statevector array types:

- `sv` is a NumPy array → uses `rng` (or a fresh default generator if `rng=None`).
- `sv` is a JAX array (e.g. the statevector returned by `circuit_to_energy_fn`'s `energy_fn`, the common case when composing noise into a `jax.grad`/`jax.value_and_grad` pipeline) → uses `jax_key` if given explicitly; otherwise, if `rng` is given, a JAX key is *derived* from it (`rng.integers(...)` seeds `jax.random.PRNGKey`) so a seeded `rng` reproduces the same sequence of keys across separate runs, the same guarantee the NumPy path already gives; only when *neither* is given does it fall back to a fresh, non-reproducible key from OS entropy.

`jax_key`, when given, always takes precedence over `rng` — pass it explicitly (and split a fresh subkey per call, e.g. `jax.random.split`) when you specifically want a different-but-reproducible noise draw each step rather than one implied by advancing a shared `rng`. Prior to v8.1.27, `rng` was silently ignored whenever `sv` was a JAX array (issue #7) — a caller who seeded `rng` expecting NumPy-style reproducibility got a different result on every call with no error or warning.

---

## ▍ Mitigation & Predictive Healing

Active error tracking and stabilization integrated natively into the simulation runtime via `healing.py`.

| Model | Operators | Description |
|---|---|---|
| `dephasing_tracking` | `Δ_pre_emp ∘ Σ` | Predictive deviation vs ideal eigenstate |
| `phi_ab_alignment` | `Φ_AB(state_A, state_B, ipg)` | Semantic + coherence alignment between two quantum states |
| `vettore_dinamico` | `V_din = K · log(E_B/E_A) · Φ_AB` | Log-differential energetic evolution vector |

All core functions compiled via `@jax.jit`. Event history managed by `MemoryReflectionEngine` with JAX Zero-Drift spectral aggregation.

### Zero-Noise Extrapolation (`dense_evolution.mitigation`)

The primitives above are building blocks, not a runnable mitigation pipeline by themselves. `mitigation.py` is the orchestrator: it exposes ZNE under the names the field already uses (`richardson_extrapolate`, `zero_noise_extrapolation`, `noise_factors`) so it's discoverable without first learning the `Δ_pre_emp`/`Φ_AB` vocabulary above — the primitives themselves are not renamed, only composed.

| Function | Signature | Description |
|---|---|---|
| `richardson_extrapolate` | `(expectation_values, noise_factors)` | Plain N-point Lagrange extrapolation to zero noise. Reduces exactly to the textbook `(3, -3, 1)` coefficients at `noise_factors=(1,2,3)` |
| `zero_noise_extrapolation` | `(expectation_values, noise_factors, sigma_at_base_noise=None, target_sigma_ideal=10.0)` | Standard entry point. Falls back to `richardson_extrapolate` when `sigma_at_base_noise` is omitted; otherwise perturbs the 3-point Richardson coefficients via `calculate_delta_preemp` — Dense-Evolution's "predictive healing" ZNE variant. Only defined for exactly 3 noise factors; raises `NotImplementedError` otherwise rather than guessing |

```python
import dense_evolution as de

e1, e2, e3 = 1.234, 0.876, 0.611          # values at 1x, 2x, 3x noise
plain = de.zero_noise_extrapolation([e1, e2, e3], [1.0, 2.0, 3.0])

# with a measured coherence signal at the base noise level, the
# extrapolation is nudged by how far it is from the ideal target:
healed = de.zero_noise_extrapolation([e1, e2, e3], [1.0, 2.0, 3.0],
                                      sigma_at_base_noise=9.3, target_sigma_ideal=10.0)
```

`richardson_extrapolate`/`zero_noise_extrapolation` accept array-valued `expectation_values` too (e.g. a full probability distribution per noise scale, extrapolated elementwise), not just scalars — this is what the Streamlit dashboard's **Mitigation (ZNE)** tab uses under the hood (`dashboard_core.mitigation_runner.run_mitigation_sweep`): run the active circuit at 1x/2x/3x the configured noise probability, extrapolate the whole probability vector (and fidelity) to zero noise, reusing these two functions exactly as-is.

---

## ▍ IA Utils — Vector Sequence Healing

`ia_utils/vector_healing.py` — standalone module for cleaning sequences of vectors (e.g. hidden states / embeddings) that may contain `NaN` or `Inf` entries. Both functions preprocess the input (Inf → NaN → column-mean imputation) before healing, so corrupted values never propagate into the output.

| Function | Approach | Returns |
|---|---|---|
| `median_healing(vettori, radius_baseline=None)` | `scipy.ndimage.median_filter`, dynamic radius `min(20, max(3, n // 3))` | `(healed: np.ndarray, radius: int)` |
| `enhanced_dense_healing_hybrid(vettori, radius_baseline=None)` | Blends the `dense_evolution.healing` Φ-trigger logic with a median fallback, decided per-step | `(healed: np.ndarray, metadata: dict)` |

`enhanced_dense_healing_hybrid` metadata:

| Key | Type | Description |
|---|---|---|
| `fallback_triggered` | `bool` | `True` only if the input contained genuine NaN/Inf corruption AND the median fallback fired to correct it -- does not reflect Phi-Trigger corrections on structurally noisy-but-valid data |
| `adaptive_radius_used` | `int` | Baseline radius actually applied |
| `reconstruction_error` | `float` | Mean norm of the correction applied vs. the sanitized input |

```python
import numpy as np
from ia_utils.vector_healing import median_healing, enhanced_dense_healing_hybrid

vettori = np.random.default_rng(0).normal(size=(50, 128))
vettori[10, 3] = np.nan          # simulate a corrupted hidden state
vettori[30, 7] = np.inf

healed, radius = median_healing(vettori)

healed_hybrid, meta = enhanced_dense_healing_hybrid(vettori)
print(meta)
# {'fallback_triggered': True, 'adaptive_radius_used': 16, 'reconstruction_error': 11.48}
```

`jax` is imported lazily inside `enhanced_dense_healing_hybrid` (a leftover from when JAX was optional) — harmless now that `dense-evolution` always installs JAX as a core dependency, but it does mean `median_healing` and the module import itself never actually needed it in the first place.

---

## ▍ Anti-OOM Chunk Engine

All operations parcellized dynamically using a 4-layer architectural shield.

| Layer | Class | Role |
|---|---|---|
| 1 | `SafeMemoryGuard` | Pre-allocation RAM check — blocks before JAX raises `RESOURCE_EXHAUSTED` |
| 2 | `MemoryChunker` | Geometry calculator — computes `num_chunks`, `chunk_dim`, `chunk_size_bits` from available RAM without any JAX allocation |
| 3 | `CircuitChunker` | Per-slice execution — `SafeMemoryGuard` fires before every gate-slice dispatch |
| 4 | `Chunk` | Top-level wrapper — logical n_qubits decoupled from physical allocation at `safe_qubits` |

### Benchmark vs PennyLane — Windows CPU (8 GB RAM)

> Dense Evolution maintains constant ~2 GB RAM at any qubit count via dynamic chunking.
> PennyLane allocates the full statevector — OOM beyond 26q.

| Qubits | Hilbert Space | PennyLane | PennyLane RAM | Dense Evolution | Dense RAM | Chunk Geometry |
|:------:|:-------------:|:---------:|:-------------:|:---------------:|:---------:|:--------------:|
| 24 | 16,777,216 | ✅ | 307 MB | ✅ | 516 MB | 1× (2²⁷) |
| 26 | 67,108,864 | ✅ | 1,074 MB | ✅ | 2,050 MB | 1× (2²⁷) |
| 28 | 268,435,456 | ❌ OOM | — | ✅ | 2,050 MB | 2× (2²⁷) |
| 30 | 1,073,741,824 | ❌ OOM | — | ✅ | 2,048 MB | 8× (2²⁷) |
| 32 | 4,294,967,296 | ❌ OOM | — | ✅ | 2,048 MB | 32× (2²⁷) |

```python
from dense_evolution import Chunk

sim = Chunk(27)
sim.run_chunk([['h', i] for i in range(27)], chunk_size_gates=500)

print(sim)
# Chunk(n_qubits=27, safe_qubits=27, num_chunks=1,
#       chunk_size_bits=27, mem_per_chunk=2048.0 MB, ram_free=42.3%, has_jax=True)
```

### Distributed dispatch across a device mesh — `run_chunk_distributed`

`run_chunk()`'s multi-chunk path (`num_chunks > 1`) solves "RAM of one process" — every chunk lives in the same machine's memory, even though the kernel that moves them is JIT-fused. `run_chunk_distributed()` solves a different constraint: "more qubits than fit on one device," with one physical chunk pinned to its own JAX device (v1 scope: `jax.device_count()` must be `>= num_chunks`, exactly one chunk per device — a hybrid multi-chunk-per-device scheme is future work).

```python
from dense_evolution import Chunk

sim = Chunk(30)  # num_chunks > 1 on most machines
sim.run_chunk_distributed([['h', i] for i in range(30)])
```

Cross-chunk gate mixing (a gate touching one of the top `m = n_qubits - chunk_size_bits` "chunk-select" qubits) becomes real point-to-point network communication — `jax.lax.ppermute`, the same pairwise-exchange pattern distributed statevector simulators use, exchanging edge data only with the fixed XOR-stride partner device, never gathering the full state onto any single device. A gate entirely within a chunk's local qubits, or a control-chunk/target-local gate (decidable from each device's own chunk index alone), needs **no** communication at all.

Requires `jax.device_count() >= num_chunks`; raises `RuntimeError` with the exact count needed otherwise. For local testing without real multi-GPU hardware, force extra simulated CPU devices via `XLA_FLAGS=--xla_force_host_platform_device_count=N` (set before the process starts — JAX's device count is fixed at first initialization, not changeable at runtime) — this validates the sharding/communication logic is correct, not real GPU-to-GPU network performance, which needs actual multi-GPU/multi-host hardware to measure honestly.

---

## ▍ Benchmarks

> Measured on Google Colab Free Tier (CPU runtime)

| Metric | Value |
|---|---|
| Numerical drift (30-layer Ansatz, 1360 gates) | `Δ = 1.11 × 10⁻¹⁶` |
| Memory footprint @ 20q | `32 MB` (float64) · `16 MB` (float32) |
| JIT compile overhead (first run) | `< 400 ms` |
| Gate throughput after warm-up | `> 10⁶ gates/s` (CPU) |
| Maximum tested qubits (Colab Free) | `24q` stable · `33q` high-RAM runtime |
| Anti-OOM latency reduction (static JIT cache) | `−86.47%` |

---

## ▍ Composer — Interactive Web UI

A real circuit editor (graphical drag-and-drop or OpenQASM 2.0), running
every circuit on the actual `DenseSVSimulator` — statevector,
probabilities, Q-sphere, Bloch spheres, a native circuit diagram (plain
matplotlib, no Qiskit `QuantumCircuit` ever constructed), plus real
molecular Hamiltonians (PennyLane Hartree-Fock), VQE, ZNE mitigation, and
a "Scan energia" panel computing a bond-length ground-state-energy curve
(e.g. a dissociation curve) in one click instead of one point at a time.
No mocked data anywhere: every number on the page comes from a real run.
Every request has a client-side timeout (30s default, 10 minutes for
VQE/MD-trajectory calls) with a clear error instead of hanging forever
indistinguishable from normal progress.

**The page is public, the compute is yours.** [tatopenn-cell.github.io/Dense-Evolution/composer](https://tatopenn-cell.github.io/Dense-Evolution/composer/)
is static — it talks to a local kernel that runs on your own machine, not
a shared server. Get the kernel running with the installer for your OS
(walks you through what it does before doing anything, no silent steps):

- Windows: [`installer/install-composer.bat`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/installer/install-composer.bat)
- macOS / Linux: [`installer/install-composer.sh`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/installer/install-composer.sh)
- Uninstall: [`installer/uninstall-composer.bat`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/installer/uninstall-composer.bat) / [`installer/uninstall-composer.sh`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/installer/uninstall-composer.sh)

The installer's only real job is `pip install dense-evolution[composer]`
plus some optional shortcuts — the kernel itself is one plain console
command, install it by hand instead if you'd rather skip the installer
(works the same from PowerShell, cmd, bash, or zsh — it's a normal Python
entry point, not a shell script):

```bash
pip install dense-evolution[composer]

dense-evolution serve                        # starts the kernel at http://127.0.0.1:8800
dense-evolution offline-composer [DEST]      # downloads the real Composer page for use with no internet
                                              # (DEST defaults to ./composer-offline)
```

`serve` is what the published page's banner is waiting for — reload the
page once it's running and the banner turns from "kernel not found" to
connected. `offline-composer` downloads the actual live page (not a
separate hand-maintained copy) plus every asset it references, keeping
the same relative layout so it opens correctly via `file://`.

---

## ▍ MCP Server — drive the Composer kernel from an agent

`dense_evolution_mcp` (`mcp_server/`) is an MCP ([Model Context
Protocol](https://modelcontextprotocol.io)) server: 21 tools, one per
Composer kernel endpoint plus a batch energy scan, a batch wormhole
sweep, and a vector-healing pass, letting an MCP-aware agent (Claude
Code, Claude Desktop, ...) run circuits, compute molecular ground-state
energies, run VQE, get QM/MM forces, run MD trajectories, apply ZNE
mitigation, run traversable-wormhole-inspired quantum teleportation, and
heal a noisy vector sequence directly — without a browser. A thin
adapter, not a reimplementation: every tool calls the same kernel
`serve` starts above; `local_site/app/server.py` itself gained three new
endpoints (`/api/wormhole_select_instance`, `/api/wormhole_teleportation`,
`/api/vector_healing`) but no existing endpoint changed.

```bash
pip install dense-evolution[mcp]

dense-evolution serve   # start the kernel first, in one terminal (see Composer above)
dense-evolution mcp     # in another terminal, or registered with your MCP client's config
```

**Register with Claude Code:**

```bash
claude mcp add dense_evolution -- dense-evolution mcp
```

- Molecule-name tools accept a short id (`"H2"`, `"LiH"`, `"HeH+"`) or the
  kernel's full descriptive catalog name — resolved from the live catalog,
  not hardcoded.
- `dense_evolution_energy_scan` computes the ground-state energy at
  several geometries in one call (the same capability the Composer page's
  "Scan energia" panel gives a human).
- `dense_evolution_wormhole_select_instance` / `_wormhole_teleportation` /
  `_wormhole_scan` run a real traversable-wormhole-inspired quantum
  teleportation protocol (Gao-Jafferis-Wall theory, arXiv:2604.10090) on a
  binary sparse SYK model — see `dashboard_core/wormhole.py` and
  `research/wormhole_syk.md` for the physics. Unlike `_energy_scan`, the
  wormhole sweep runs sequentially, not concurrently (each call is a real
  multi-second simulation; concurrent calls were found to crash the
  kernel via a BLAS/eigh thread-safety issue).
- `dense_evolution_vector_healing` reintegrates the predictive-healing
  engine (`dense_evolution.healing`'s Phi-Trigger primitives, via
  `ia_utils.vector_healing.enhanced_dense_healing_hybrid`) that shipped
  with the pre-rebuild dashboard_core's Streamlit "AI healing shield"
  middleware, left behind (not removed) when dashboard_core was rebuilt
  around the Composer kernel — see `dashboard_core/vector_healing.py`.
  Cleans a noisy (n_steps, dim) sequence (VQE telemetry, MD trajectory,
  or any other vector sequence): per step, keeps genuine dynamics,
  replaces static noise with the local median, always sanitizes NaN/Inf.
- Large arrays are truncated to their most significant entries, and
  images are saved to disk (path returned) rather than inlined as base64,
  so a tool response stays usable in an agent's context.
- `mcp_server/README.md` has full setup details; a project-scoped Claude
  Code skill (`.claude/skills/dense-evolution-mcp/`) documents how to use
  the tools effectively — which tool for which task, known cost/qubit
  caps, why arrays are truncated.

---

## ▍ Dashboard Panels

| Panel | Contents |
|---|---|
| **Overview** | R0 header · R1 P(\|n⟩) histogram + Top-12 states · R2 wavefunction helix 3D + metrics table · R3 noise analysis + shot histogram · R4–R6 VQE telemetry ×3 · R7 Pearson heatmap |
| **Fisica Stato** | Bloch projection · Schmidt rank · coherence vector |
| **Mosaico** | 2D probability density map up to 1008 qubits |
| **VQE Results** | 6-subplot: energy convergence, entropy, purity, ‖∇L‖, noise factor, θ-correction |
| **MD Results** | 6-subplot MD telemetry + masked Pearson correlation heatmap |
| **Performance** | Gate throughput · JIT compile time · RAM usage |

---

## ▍ VQE Engine

Built on `circuit_to_energy_fn` (see previous section) — no separate mechanism. Parameter injection:
1. Counting parametric gates (`rx ry rz p u1 cp crz`) → `n_params`
2. Initializing `θ ∈ ℝⁿ` uniform in `[−π, π]`
3. Injecting `θ[i]` sequentially by gate order, via a `-1.0` sentinel in the compiled op template patched in with `jnp.where` inside a `jax.lax.scan` — never a Python `float()` call, which would sever the JAX trace and make the gradient below fake.

Compatible with any custom OpenQASM 2.0 string without pre-labelling.

**Gradient & update rule:**

$$\frac{\partial E}{\partial \theta_i} = \left\langle\psi(\theta)\left|\frac{\partial H}{\partial \theta_i}\right|\psi(\theta)\right\rangle \qquad \theta \leftarrow \theta - \frac{\alpha\,\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$

**Telemetry columns** (→ `df_vqe_telemetry`):

| Column | Unit | Description |
|---|---|---|
| `VQE_Energy` | Ha | ⟨ψ\|H\|ψ⟩ |
| `Entropy` | bit | −Tr(ρ log₂ ρ) |
| `Purity` | — | Tr(ρ²) ∈ [1/d, 1] |
| `Gradient` | — | ‖∇L‖ — barren plateau detection |
| `Noise_Factor` | — | Fidelity-derived noise proxy |
| `Theta_Correction` | rad | ADAM step norm |

---

## ▍ Hamiltonian Library

Real Hartree-Fock + fermion-to-qubit mapping (PennyLane `qchem`, Jordan-Wigner
or Bravyi-Kitaev — both represent the identical physical Hamiltonian, just a
different qubit basis), exact dense diagonalization for the ground-state
energy. `dashboard_core.MOLECULE_CATALOG`:

| Molecule | Qubits | Bond length | E₀ (Ha, Jordan-Wigner) |
|---|:---:|:---:|:---:|
| H₂ | 4 | 0.7414 Å | −1.1373 |
| HeH⁺ | 4 | 0.7743 Å | −3.0157 |
| H₃⁺ (D3h triangle) | 6 | 0.8738 Å | −1.2973 |
| LiH | 12 | 1.5949 Å | −7.8824 |
| H₂O (104.5°, frozen-core O 1s) | 12 | 0.9584 Å | −75.0127 |

Custom molecules: any atomic symbols + `[[x, y, z], ...]` geometry in
Ångström, same Hartree-Fock pipeline — capped at 12 qubits (exact dense
diagonalization's real limit here, rejected with a clear error before
PennyLane runs rather than attempted and failing expensively).
`dense_evolution_energy_scan` (MCP) / the Composer's "Scan energia" panel
compute this at several geometries in one call, e.g. a dissociation curve.

---

## ▍ Circuit Library (20 presets)

All circuits stored as OpenQASM 2.0 strings in `dashboard_core.QASM_LIBRARY`.

**States** — Bell state (2q), GHZ (3q/4q/8q), W state (3q), Superposition (1q)

**Entangling layers** — linear/ring/full/star/brick topology, 5q each (`dense_evolution.topology.entangling_layer`)

**Algorithms** — Quantum Fourier Transform (3q), Grover search (3q, target `|111⟩`), Deutsch-Jozsa (2+1q, balanced oracle), Bernstein-Vazirani (3+1q, secret `101`), Toffoli/CCX (3q, T-gate decomposition), Quantum Teleportation (3q, deferred-measurement principle — verified fidelity 1.0000000000 across 5 random states), Quantum Phase Estimation (3-qubit counting register on a T-gate — verified probability 1.0000 on the exact expected phase, not merely the most likely outcome)

**Testing** — Random circuit, fixed seed (3q and 5q variants)

---

## ▍ Changelog

### v8.1.49
- **New: real traversable-wormhole-inspired quantum teleportation** (Gao-Jafferis-Wall theory, `arXiv:2604.10090`), on a binary sparse Sachdev-Ye-Kitaev (SYK) model -- reproduced as an actual verified simulation, not decoration: an earlier, discarded `dashboard_core` circuit used the right vocabulary (SYK scrambling, a phase "kick") but ran on a single qubit register, which the no-signaling theorem forbids from ever showing the protocol's real sign-dependent signature, verified directly (identical results for either sign of the kick). The real recipe needs two coupled chaotic systems, a message injected via a separate reference-qubit pair, a real bilinear L-R coupling, and a mutual-information readout (not a single-qubit expectation value) -- see `research/wormhole_syk.md` for the full derivation and every verification step (Majorana anticommutation, SYK Hermiticity, Bell-pair/GHZ mutual information, the paper's own instance-selection criterion, Trotter-vs-exact convergence).
  - **New `dense_evolution` modules**: `fermions.py` (`majorana_pauli_terms` -- Majorana-fermion → qubit Jordan-Wigner mapping), `entropy.py` (`partial_trace`, `von_neumann_entropy`, `mutual_information` -- multi-qubit, MSB-first, distinct from `dashboard_core/state_visuals.py`'s older single-qubit little-endian version), `trotter.py` (`pauli_rotation_ops`, `trotter_evolve_ops` -- real-time Hamiltonian evolution as an actual gate circuit, verified to fidelity 1.0 against `scipy.linalg.expm` and to converge smoothly to exact evolution as Trotter step count increases). Generic building blocks, not specific to the wormhole experiment -- promoted from the original `research/wormhole_syk.py` reproduction once verified, with real tests (`tests/test_fermions.py`, `tests/test_entropy.py`, `tests/test_trotter.py`) checked against known-exact cases (anticommutation relations, textbook Bell-pair/GHZ mutual information, exact `expm`), not just import smoke tests.
  - **New `dashboard_core.wormhole`**: the SYK/wormhole-specific logic (`build_sparse_syk_terms`, `commuting_pair_count`, `select_good_instance`, `run_wormhole_protocol`, `run_wormhole_protocol_trotter`) that stays genuinely specific to this experiment, built on the promoted `dense_evolution` utilities above -- mirrors how `dashboard_core.hamiltonians`/`vqe` are structured.
  - **New Composer kernel endpoints**: `POST /api/wormhole_select_instance`, `POST /api/wormhole_teleportation` (`backend='exact'|'trotter'`) -- both real, tested end to end against a live kernel, reproducing the exact reference values from the original research reproduction (seed 61, t0=0.3, t1=0.60: I(mu=+12)=0.01326/0.01301 exact/Trotter, I(mu=-12)=0.01793/0.01821).
  - **New `dense_evolution_mcp` tools**: `dense_evolution_wormhole_select_instance`, `_wormhole_teleportation`, `_wormhole_scan` (batch `t1` sweep, both mu signs, matching the `_energy_scan` pattern). Bringing the tool count to 20.
  - **Real bug found and fixed**: `dense_evolution_wormhole_scan`'s first implementation ran every sweep point (and both mu signs within a point) concurrently via `asyncio.gather`, mirroring `_energy_scan`'s pattern -- but this protocol's per-call cost is far higher (real exact diagonalization or a real Trotterized circuit over the full L+R+P+Q system, several seconds per call, not a cheap Hamiltonian lookup), and concurrent calls crashed the kernel process outright with a Windows access-violation inside concurrent `numpy.linalg.eigh` calls (a BLAS thread-safety issue under this protocol's heavier linear algebra). Found by actually running the batch tool's tests, not assumed safe by analogy to `_energy_scan`. Fixed by running sweep points sequentially; documented in the tool's own docstring so a sweep's real wall-clock cost (several minutes for 20 points) isn't a surprise.
  - 26 new `dense_evolution` unit tests, 14 new kernel/MCP end-to-end tests (all against the real, live kernel/simulator -- no mocked physics), pinned against the exact numeric reference values established during the original research reproduction.

### v8.1.48
- **New: `dense_evolution_mcp`, an MCP (Model Context Protocol) server for the Composer kernel** (`mcp_server/`, `dense-evolution[mcp]` extra, `dense-evolution mcp` console subcommand) -- 17 tools, one per `local_site/app/server.py` endpoint plus `dense_evolution_energy_scan`, letting an MCP-aware agent (Claude Code, Claude Desktop, ...) drive circuit runs, molecular Hamiltonians, VQE, QM/MM forces, MD trajectories, and ZNE mitigation directly, without a browser. A thin adapter, not a reimplementation -- every tool calls the same kernel the published Composer page uses; `local_site/app/server.py` itself is untouched.
  - Molecule-name tools accept either a short id (`"H2"`, `"LiH"`, `"HeH+"`) or the kernel's full descriptive catalog name, resolved from the live catalog rather than hardcoded -- found to matter after actually using the adapter for a real task (an H2 dissociation curve), where reproducing the full `"H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]"`-style string across several tool calls was itself a source of friction.
  - **New: `dense_evolution_energy_scan`** -- ground-state energy at several geometries (e.g. a bond-length dissociation curve) in one call instead of one per point; points run concurrently, and a failing point (e.g. one needing too many qubits) is reported with its own error instead of aborting the rest of the scan.
  - Large arrays (statevector/probabilities) are truncated to their most significant entries, and circuit/histogram/Q-sphere/Bloch images are saved to disk and returned as file paths rather than inlined as base64, so a tool response stays usable in an agent's context instead of flooding it with a wall of numbers or an unreadable image blob.
  - **Real bug found and fixed**: `mcp` 2.0.0 (released days before this work, now what `pip install mcp` gets unpinned) renamed `FastMCP` to `MCPServer` and moved it from `mcp.server.fastmcp` to `mcp.server.mcpserver` -- passed every local test regardless because the dev environment had the older 1.28.1 cached, and only surfaced on CI's clean install (`ModuleNotFoundError: No module named 'mcp.server.fastmcp'`). Fixed by migrating the two affected import lines; confirmed directly against a real 2.0.0 install that the rest of the API (`@mcp.tool(name=..., annotations={...})` with a plain dict, Pydantic model parameters, `mcp.run()`, decorated functions staying directly callable) is unchanged between the two majors.
  - 24 new tests (`tests/test_mcp_server.py`), 85.86% coverage on `mcp_server/server.py` (Codecov `codecov/patch` on this release: 100% of the diff) -- routed through `httpx.ASGITransport` straight into the real, in-process kernel app rather than a live subprocess, so every test still exercises the real `DenseSVSimulator`/PennyLane Hartree-Fock, no mocked physics, the same principle `test_local_site_server.py` already used for the kernel itself.
- **Composer page**: added request timeouts (`api()` had none at all -- a genuinely slow or stuck request left "...in corso" on screen forever, indistinguishable from normal progress; 30s default, 10 minutes for VQE/MD-trajectory calls, matching the "can take minutes" warning already shown for those) and a "Scan energia" panel computing a bond-length ground-state-energy curve in one click, reusing the existing geometry-generator helper instead of requiring "Compute ground state" to be clicked once per point by hand -- the same dissociation-curve gap fixed above for the MCP adapter, now fixed for human visitors too. Verified in a real headless-Chromium session (Playwright) against a live kernel: connects, computes a real 5-point curve with the correct minimum, button disables while running and re-enables after, validation errors display correctly, and a real VQE run still completes with no console errors.

### v8.1.47
- **`run_circuit_jit` / `run_batch_jit` renamed** (from `run_circuit_jit_beast_mode` / `run_parametric_batch_jit`) is now on PyPI -- this was already on `main`/GitHub since right after v8.1.46 but held back from release; batched in here. The old names still work, emitting a `DeprecationWarning`.
- **New: `dense_evolution.harrison_tb`** -- dependency-free (numpy only) sp3 tight-binding Hamiltonian builder using Harrison's *universal* parameter table (one fixed set of 4 eta coefficients for every element pair, atomic term values from Harrison's Solid State Table). Builds single-bond dimers (`sp3_dimer_hamiltonian`) and full periodic zinc-blende crystals via Bloch sums (`zincblende_hamiltonian`) -- no PySCF/OpenFermion/SCF anywhere. Zero per-material setup cost, at the price of accuracy (real-material validation: [Dense-Evolution-Ising-Tests](https://tatopenn-cell.github.io/Dense-Evolution-Ising-Tests/harrison_tight_binding/)).
- **New: `dense_evolution.vhd_tb`** -- sp3s* tight-binding using Vogl-Hjalmarson-Dow (1983) *material-specific* fitted parameters (16 zinc-blende/diamond semiconductors), far more accurate than `harrison_tb`'s universal table at the cost of a fitted parameter row per material. `band_extrema_along_path` scans a k-space line to find the true valence-band max / conduction-band min for indirect-gap materials (Si, Ge), where the minimum sits off-Gamma and reading only Gamma gives the wrong number. Found and fixed a Hermiticity bug in the reference implementation ported from while building this (`Hac.conjugate()` without transposing).

### v8.1.46
- **Composer ships**: the interactive web UI (`feature/composer`) merged into `main` and is now published at [tatopenn-cell.github.io/Dense-Evolution/composer](https://tatopenn-cell.github.io/Dense-Evolution/composer/) -- a graphical/OpenQASM circuit editor running every circuit on the real `DenseSVSimulator`, plus real molecular Hamiltonians, VQE, and ZNE mitigation. The page is static; it talks to a local compute kernel (`local_site/app/server.py`) the visitor runs on their own machine via `pip install dense-evolution[composer]` -- never a shared server, nothing sent anywhere else.
- **Every dashboard_core visualization is now Qiskit-free**, not just optional-without-it: the circuit diagram, histogram, Q-sphere, and Bloch spheres are all native matplotlib/numpy (`dashboard_core.circuit_diagram`, `dashboard_core.state_visuals`). This closes three real macOS segfault sites v8.1.44's fix didn't cover (the Circuit-diagram panel still built a `QuantumCircuit` there) and removes `qiskit`/`streamlit` as hard dependencies of the Composer kernel entirely -- both were previously imported at module level in `dashboard_core`, so the kernel failed to even start without them installed, discovered by actually running the installer on a clean machine. Q-sphere/Bloch spheres are computed directly from the statevector's own reduced density matrices (partial trace), verified against known physics (a Bell pair's each qubit comes back exactly maximally mixed) and performance-tested up to 24 qubits (worst case ~7s, vs. `qiskit.visualization.plot_bloch_multivector`'s ~252s at just 12 qubits -- the reason that panel needed a qubit-count skip in the first place, now removed).
- **Two new real memory-safety gaps closed**: `dashboard_core.mitigation`'s two ZNE panels and `dashboard_core.hamiltonians.build_molecular_hamiltonian`'s dense diagonalization never checked `SafeMemoryGuard` before allocating (unlike `engine.py`, which always did) -- the density-matrix ZNE variant and the Hamiltonian's `eigvalsh` are both quadratically worse than a plain statevector at the same qubit count, and the Composer's own geometry generators let a visitor build an arbitrarily long atom chain with no smaller natural ceiling. Both now guard, sized for what they actually allocate.
- **`dense-evolution` console script + `[composer]` extra**: `dense-evolution serve` starts the local kernel; `dense-evolution offline-composer [DEST]` downloads the real published Composer page (not a hand-maintained copy) for use with no internet, preserving its real relative asset layout. Full installers for Windows/macOS/Linux (`installer/`) install the kernel, optionally download the offline copy, and optionally add Desktop/Start Menu-or-app-menu/autostart shortcuts for both the online and offline entry points -- reversible via matching uninstallers. All of it verified by actually running the installer end-to-end on a real machine (not just reading the code): found and fixed a real bug in the offline downloader (it tried to fetch navigation `<link>` tags pointing at other pages, not real assets) this way.

### v8.1.45
- **`dashboard_core.vqe.run_vqe` now optimizes both ansatz families entirely on `dense_evolution`'s own engine** -- `dense_evolution.autodiff.circuit_to_energy_fn` (JAX-differentiable) plus a hand-rolled Adam loop, no PennyLane device/QNode/optimizer involved in the optimization itself anywhere in this module. Hardware-efficient was a direct fit (its RY/CX ansatz is plain OpenQASM). UCCSD was not: PennyLane's own decomposition of `qml.UCCSD` reuses each of the molecule's few real weights across several RX/RZ gates per excitation (the Trotter exponentiation of that excitation's Pauli-string terms), while `circuit_to_energy_fn` treats every parametric gate occurrence as an independent free parameter. Solved with an affine parameter expansion (`full_gate_values = baseline + expansion_matrix @ real_weights`) derived by directly probing PennyLane's own decomposition at zero and at each basis weight vector -- verified to reproduce PennyLane's own reported energy to floating-point noise (2.5e-14) for arbitrary weight vectors, and composed with `circuit_to_energy_fn` this stays JAX-differentiable in the small real weight vector by ordinary chain rule, so the same Adam loop optimizes it. PennyLane's only remaining role anywhere in this module is the real Hartree-Fock + Jordan-Wigner mapping itself (`dashboard_core.hamiltonians`), not something worth reimplementing. Re-verified end-to-end against H2: hardware-efficient converges to -1.13726 Hartree (exact -1.13727), UCCSD to 6e-6 Hartree from exact.

### v8.1.44
- **Added**: `pauli_hamiltonian_to_matrix(terms, n_qubits)` -- builds the real dense Hermitian Hamiltonian matrix for a weighted sum of Pauli strings (`sum_i coeff_i * P_i`, each `P_i` a Kronecker product of per-qubit Pauli matrices), the explicit-matrix counterpart to `pauli_sum_expectation` (which deliberately never builds it). Verified against `qml.matrix()` for H2/HeH+/H3+/LiH (identical matrix and ground-state energy, not just close) and against an independent brute-force `numpy.kron` reference across 50 random multi-term Hamiltonians.
- **Reintegrated `dashboard_core`** onto `main` (previously moved off for a ground-up rebuild) with real molecular quantum chemistry built on top of it: real molecular Hamiltonians (PennyLane Hartree-Fock, Jordan-Wigner/Bravyi-Kitaev mapping -- verified spectrum-identical, as it must be -- catalog of H2/HeH+/H3+/LiH/H2O, H2O using a real frozen-core active space to stay at 12 qubits), real VQE (hardware-efficient and UCCSD ansätze, Adam + adjoint differentiation on `lightning.qubit`; UCCSD's decomposed circuit verified to reproduce PennyLane's own reported energy to 1e-14 when executed independently on `dense_evolution`), real Hamiltonian mixing (weighted sum of two same-qubit-count molecules), and real Zero-Noise Extrapolation (both scalar Pauli-expectation and density-matrix variants, each an ensemble average over many stochastic noise-channel draws, not a single trajectory).
  - **Real bug found**: `dashboard_core.engine.run_circuit_from_qasm` and both functions in `dashboard_core.mitigation` called `QuantumCircuit.from_qasm_str` -- a second, independent instance of the macOS `QuantumCircuit` segfault documented in v8.1.43, this time in a genuine production code path (the Composer's Circuit-diagram panel and its two ZNE panels), not just a test. Fixed two different ways depending on whether a real Qiskit circuit object was actually needed downstream: `engine.py` still needs one (for the diagram), so it now parses with `dense_evolution`'s own `QASMParser` and builds the `QuantumCircuit` from the parsed gate tuples via plain method calls, never touching `qasm2`; `mitigation.py` never needed a Qiskit object at all, so it now runs straight off `QASMParser` + `DenseSVSimulator.run_circuit`, removing the risk entirely rather than working around it.
  - **Real gap found**: CI never installed `pylatexenc`, required by `circuit.draw(output='mpl')` -- failed identically on every OS (Ubuntu included), unrelated to the macOS issue above.
  - 41 new tests (`test_dashboard_hamiltonians.py`, `test_dashboard_vqe.py`, `test_dashboard_mitigation.py`, `test_dashboard_system_limits.py`) closing the coverage gap Codecov flagged right after reintegration (`vqe.py` 7.6%, `hamiltonians.py` 22.0%, `mitigation.py` 35.4%, `system_limits.py` 42.9%) -- real molecular ground-state energies checked against known values, real VQE convergence for both ansatz families, real noise-channel decay and fidelity-correction behavior, not mocks.
  - The Composer web app itself (`local_site/`, FastAPI + vanilla JS, the interactive frontend built on this `dashboard_core`) stays off `main` on `feature/composer` -- not yet ready to ship alongside the package release.

### v8.1.43
- **Added**: seven standard-convenience functions Qiskit/PennyLane ship as everyday API that this package didn't: `entangling_layer(n, pattern='linear'|'circular'|'full'|'star'|'brick')`, `pauli_expectation`/`pauli_sum_expectation` (Pauli-string expectation values via O(dim) bit manipulation, never building the 2^n_qubits Hamiltonian matrix), `ghz_state(n)`, `sample_counts(sv, n_shots)` (Qiskit-style `get_counts()`), `statevector_fidelity(a, b)` (the pure-state counterpart to `uhlmann_fidelity`, which only handles density matrices), `qft(n, inverse=False, do_swaps=True)`, `random_circuit(n_qubits, n_gates, ...)`, and `draw_circuit(circuit, n_qubits)` (plain-text diagrams). The first five were confirmed as real gaps, not speculative additions: the same patterns were found hand-duplicated with no shared, tested implementation across 20+ VQE/observable scripts and every test/experiment that preps a GHZ or Bell state, in this repo and the sibling `Dense-Evolution-Ising-Tests`. `qft`/`random_circuit`/`draw_circuit` round out the pass for completeness (no internal duplication motivated them the same way).
  - **Real bug found and fixed while verifying `pauli_expectation`**: the first implementation assumed qubit `q` is bit `q` of the basis-state index. `DenseSVSimulator` actually stores qubit 0 as the *most significant* bit (confirmed empirically and matches `conftest.py`'s own patched `measure()`: `phys_q = self.n - 1 - qubit_idx`). Fixed and re-verified — 500 random 4-qubit states cross-checked against brute-force dense Pauli matrices (`numpy.kron`), exact match (max error 0.00e+00). `qft` was verified the same way: max error 1.4e-15 against the analytic DFT matrix across 1-4 qubits, plus an exact QFT-then-inverse-QFT round trip.
  - **Real bug caught before commit**: `draw_circuit`'s first version used Unicode box-drawing characters (`─│●`), which crashed `print()` on a default Windows console (cp1252 can't encode them). Switched to plain ASCII (`-|*`) — a diagram meant for a terminal or log file has to survive whatever console encoding the caller has.
  - 72 new tests across `test_topology.py`/`test_observables.py`/`test_states.py`/`test_measurement.py`/`test_qft.py`/`test_random_circuit.py`/`test_drawing.py`.
- **Added**: `macos-latest` to the CI test matrix (previously `ubuntu-latest` only), prompted by a direct question about macOS support that had never actually been verified. This surfaced two real, unrelated macOS-only process crashes, both fixed:
  - **Qiskit itself segfaults on macOS CI runners.** `qiskit.circuit.QuantumCircuit.__init__` — the simplest possible call, `QuantumCircuit(3)` — reproducibly crashed the whole process (SIGSEGV) on Python 3.10/3.11/3.12, macos-latest (arm64), deterministically at the same line every time. Not a Dense-Evolution bug: every Dense-Evolution-only test passed cleanly on macOS before hitting this file. First fix attempt (`skipif` on the test class) didn't actually solve it — `qiskit = pytest.importorskip('qiskit')` is a class-body statement, and pytest runs class bodies at collection time regardless of a skip marker on the class, so qiskit was still being imported into the process either way, and the crash just moved to a non-deterministic segfault during interpreter shutdown instead. Real fix: guard the class *definition* itself behind `if sys.platform == 'darwin'`, so the class body never executes there at all; a `pytest.mark.skip`-decorated empty stub stands in for visibility. `TestPennyLaneInterop`, same file, unaffected either way.
  - **A second, unrelated shutdown-time segfault, independent of Qiskit.** Confirmed by the above fix: with Qiskit fully removed from the macOS process, the exact same class of crash (SIGSEGV, exit 139) still happened on Python 3.10/macos-latest — every one of 523 tests passing, `coverage.xml` already written, pytest's own summary already printed, then a crash a few seconds later during Python interpreter finalization. Points to native-extension teardown (most likely JAX/XLA's runtime shutdown on macOS ARM, a known category of issue unrelated to this package's own code). Fixed with a `trylast` `pytest_sessionfinish` hook in `conftest.py` that calls `os._exit(exitstatus)` on macOS right after pytest's own work — and pytest-cov's coverage.xml write — are done, bypassing the interpreter-finalization phase that was crashing. No-op on Windows/Linux.
  - CI now runs 6 jobs (2 OSes x 3 Python versions) instead of 3; all green.

### v8.1.42
- **Added**: a full documentation site, [tatopenn-cell.github.io/Dense-Evolution](https://tatopenn-cell.github.io/Dense-Evolution/) — MkDocs + Material, auto-deployed to GitHub Pages on every push to `docs/**`/`mkdocs.yml`/`dense_evolution/**` via `.github/workflows/docs.yml`. The full API reference (`api/*.md`, 11 module pages) is generated directly from this codebase's own docstrings via `mkdocstrings` (`docstring_style: google`), not hand-duplicated; the Changelog and License pages are single-sourced from this README and `license.md` via `mkdocs-include-markdown-plugin` so they can't drift out of sync. Two real bugs surfaced only by the strict-mode CI build, both fixed before the first deploy: a docstring bracket sequence in `QASMCircuit.to_tuples` misparsed as a broken markdown link by `mkdocs-autorefs`, and a Linux/Windows filesystem case-sensitivity mismatch (`docs/LICENSE.md` including `../LICENSE.md`, but the tracked file is lowercase `license.md`) that only failed on the Linux CI runner, not locally.
  - **First honest self-review pass** (rereading the live site page by page rather than assuming it was finished) found and fixed 6 real gaps: the Getting Started Zero-Noise-Extrapolation example referenced undefined variables and couldn't actually be run as published; the dashboard command didn't mention `app_dashboard.py` only exists in a cloned checkout, not the pip package; there was no examples/tutorial page beyond the short Quick Start (added `docs/examples.md` — density-matrix ZNE healing, MPS for low-entanglement circuits, differentiable VQE, each adapted from already-tested code — `experiments/matrix_healing_zne.py`, `test_mps.py`, `test_autodiff.py` — not written fresh); related API pages had no cross-links (`mitigation.md` ↔ `healing.md`/`registry.md`, `mps.md` ↔ `chunk.md`/`simulator.md`); there was no favicon/logo (hand-authored `docs/assets/favicon.svg`, a qubit-orbit glyph in the site's own cyan accent); and there was no architecture diagram (added a Mermaid module-dependency graph to `docs/index.md`, traced from the real `import` statements in every `dense_evolution/*.py` file, not an idealized layering).
  - **Second review pass** found 3 more: API reference function signatures rendered as cramped single-line text because `mkdocstrings` needs Black or Ruff installed to format them (added `ruff` to the `docs` extras — confirmed via a before/after strict rebuild that signatures now render properly indented); this README's own `## ▍ Benchmarks` section (real throughput/PennyLane-comparison numbers) had no path onto the site at all (added `docs/benchmarks.md`, single-sourced from this file); and `CONTRIBUTING.md` existed in the repo but wasn't linked from the site nav (added `docs/contributing.md` — also fixed two of its relative links, to `license.md`/`SECURITY.md`, to absolute GitHub URLs so they resolve correctly both on GitHub and once included into the docs site, the same class of case-sensitivity fragility as the earlier LICENSE.md bug).
  - **Added**: `test_docs_examples.py`, run in CI (`ci.yml`) on every push — locates each Python code block actually published in `getting-started.md`/`examples.md` by its section heading and executes it directly (not a hand-maintained copy), asserting on real output (ZNE fidelity improves, GHZ probabilities are exactly 0.5/0.5, VQE converges well below the Hamiltonian's mid-spectrum). A future signature change anywhere these examples touch now fails CI immediately instead of silently going stale until someone rereads the site by hand. Also brought `CONTRIBUTING.md`'s own "Running tests" command back in sync with what CI actually runs — it had already drifted, missing four test files, before this release.
  - This README now links the docs site prominently (a `docs` badge alongside the existing CI/PyPI badges, plus a callout line right under "What It Is").

### v8.1.41
- **Added**: test coverage tracking via `pytest-cov` (`[tool.coverage.run]`/`[tool.coverage.report]` in `pyproject.toml`, a `dev` extras group, a coverage step in CI uploading `coverage.xml` as an artifact -- codecov badge/upload would need a `CODECOV_TOKEN` the user sets up themselves, not done here). Real measured project-wide coverage, honestly characterized before writing anything: an initial pass found 84.1%, driven up to **94.4%** (449/450 tests passing, one isolated failure confirmed to be the same RAM-pressure artifact already documented for `TestChunkMultiPiece`, not a regression) by closing genuinely untested code paths across `dashboard_core/md_telemetry.py` (39.8%→100%), `dashboard_core/qasm_library.py` (61.1%→100%), `dashboard_core/simulation_runner.py` (64.4%→87.6%), `dense_evolution/chunk.py` (63.6%→95.9%), `dense_evolution/registry.py` (84.9%→99.5%), and `dense_evolution/parser.py` (87.6%→97.4%). `dense_evolution/simulator.py` (72.5%→78.6%) is limited by a confirmed coverage-tooling artifact, not a real gap: `DenseSVSimulator.measure()` is already thoroughly tested (`TestMeasurement`, 5 pre-existing tests) but `coverage`/`pytest-cov` fails to trace it correctly inside this specific large test suite -- confirmed directly with an isolated `coverage.Coverage()` script (bypassing pytest entirely) that the method traces correctly on its own, so no duplicate tests were added for it. `dashboard_core/interactive_panel.py` (82.0%) was deliberately left as-is, out of scope -- it overlaps with the separate, larger, not-yet-started "rebuild `launch_interactive_panel` with full Streamlit parity" task.
  - **Real bug found and fixed along the way**: `dashboard_core.simulation_runner.estrai_valore_puro` silently returned `0` for *any* numeric string input (e.g. `estrai_valore_puro("3.5")` returned `0`, not `3.5`) -- every Python `str` has a `.index()` *method*, so `hasattr(elemento, 'index')` was true for any string, routing it into the "object with an `.index` attribute" branch before ever reaching the `isinstance(str)` numeric-parsing block below it; calling `.index()` with no arguments always raised `TypeError`, silently caught and falling through to `0`. If the QASM parser ever handed this function a string-typed qubit index or parameter, it was silently read as `0` instead of the real value. Fixed by moving the `isinstance(str)` check to run first.
  - `chunk.py`'s biggest apparent gap (~200 lines, the distributed multi-device kernel) turned out to be a measurement gap, not a test gap: it's already covered by `TestChunkDistributed`, which only runs under `XLA_FLAGS=--xla_force_host_platform_device_count=8` in a separate pytest invocation (same as CI already does) -- combining both invocations' coverage data gave the real 95.9% number.
  - 61 new tests total across `test_dense_evolution.py` and `test_dashboard_core.py`.

### v8.1.40
- **Added**: full `jax.jit`-compatible coverage for every function in `dense_evolution.mitigation` -- `richardson_extrapolate_jit`, `zero_noise_extrapolation_jit` (the predictive-healing branch), `polynomial_extrapolate_jit`, `uhlmann_fidelity_jit`, `zne_density_matrix_jit`. Each is a jit-safe `_core` (no `np.iscomplexobj`/`np.asarray`/`float()` calls on possibly-traced values -- those break `jax.jit` tracing) plus an unchanged eager wrapper, the same split already used by `dense_evolution.mps`'s `_jsd_vectors_jax`/`_jsd_vectors`. Verified: each `_jit` variant matches its eager counterpart exactly; all compose correctly together inside a *single* outer `jax.jit` (the realistic use case -- several of these called in sequence inside a step function passed to `jax.lax.scan`, not each jitted in isolation). Real measured speedup on the full `zne_density_matrix` pipeline (not a single function): 4.5x-150x across 2x2-32x32 matrices, positive in every case tested.
  - Checked and rejected a suggestion (from the same external-AI source as v8.1.39's rewrite) to mark `target_sigma_ideal` static via `functools.partial(static_argnames=...)`: `calculate_delta_preemp` already uses `jnp.where` internally, not a Python `if`, so it's trace-safe as a dynamic value -- marking it static would force a fresh XLA recompilation every time a caller varies it, with no benefit. Confirmed directly by testing both ways before deciding, not by assumption.
  - This closes out the density-matrix ZNE healing work for now. A further step -- wiring these functions into `MPSSimulator` so error mitigation can run on tensor-network-simulated circuits -- was scoped and intentionally deferred: it requires noise-channel support for MPS (doesn't exist yet, a large feature on its own -- MPO-based or trajectory-based), reduced (not global) density-matrix extraction via partial trace (also doesn't exist yet), and handling the multiple-noise-scale requirement ZNE needs, none of which reduce to "just call the existing functions." A global density matrix is the wrong target for large-qubit MPS use in the first place -- materializing a full 2^n x 2^n matrix defeats MPS's own reason for existing. Tracked separately, not started here.

### v8.1.39
- **Changed**: `project_to_physical`/`uhlmann_fidelity` rewritten as fully vectorized, `jax.jit`-compatible JAX (no Python `while`/`for` loop over eigenvalues) -- the previous versions had a dynamic Python loop that isn't traceable, forcing a host round-trip on every call. `project_to_physical` now uses Euclidean projection onto the probability simplex (Held, Wolfe & Crowder 1974; also Duchi et al. 2008) applied to the eigenvalues -- a different, fully array-vectorized algorithm for the exact same convex projection problem Smolin-Gambetta-Smith's paper solves (unique global minimum, so any correct algorithm must agree). `uhlmann_fidelity` now computes `Tr(sqrt(inner))` as the sum of `inner`'s eigenvalues' square roots instead of reconstructing the full matrix square root. Both verified numerically identical to the previous versions (~1e-15 max difference on the SGS paper's own worked example and 30 random test matrices) and confirmed to actually compile and run under `jax.jit`. No behavior change for any existing caller -- same inputs, same outputs, just usable inside a jitted pipeline (e.g. `jax.lax.scan`) going forward.
  - Originated as a suggestion from an external AI (Gemini), independently verified numerically before adopting -- same source also proposed reintroducing the predictive-healing coefficient perturbation (`calculate_delta_preemp`) for the density-matrix case as "the real improvement." That specific claim was rejected: it directly contradicts the already-measured result that the perturbation's effect is negligible (~0.14% coefficient shift even at a large observed coherence deviation, confirmed with a live numeric trace, not assumed) -- an external source getting a plausible-sounding architecture story right doesn't make an unverified empirical claim inside it right; each part was checked on its own.

### v8.1.38
- **Documented**: `zne_density_matrix`'s docstring extended with a broader honest finding beyond v8.1.37's original single-configuration result. `experiments/matrix_healing_zne_sweep.py` (new) sweeps GHZ states from 2 to 5 qubits across all 5 `NoiseModel` channels (depolarizing, bitflip, phaseflip, amplitude_damping, combined). First pass (3 seeds, K=150 trajectories, 60 runs) looked mixed -- 54/60 positive, mean delta +0.09, but apparently unreliable for phaseflip/amplitude_damping (small or net-negative deltas at some qubit counts). Investigated further before trusting it: `richardson_extrapolate`'s 3-point Lagrange coefficients (3, -3, 1) amplify statistical noise in their inputs (sum of squares 19x a single raw measurement's variance), so an undersampled density-matrix estimate makes the *corrected* result noisy even when the correction itself is sound -- confirmed directly by re-running the same "failing" configurations at higher K (300-1200 trajectories, more seeds), which turned them consistently and strongly positive. Re-ran the full sweep properly (K=400, 5 seeds, 100 runs total): **96/100 positive, mean delta +0.12**, every single (qubit count, noise channel) combination net positive on average, growing to +0.20-0.25 at 5 qubits for depolarizing/bitflip. The 4 remaining negative runs are small (worst -0.02), consistent with residual Monte Carlo noise, not a systematic failure. No code behavior changed -- this is a documentation update, but the earlier "unreliable for phaseflip/amplitude_damping" wording in an intermediate draft of this entry was itself wrong and has been corrected here before ever reaching PyPI.
  - `zne_density_matrix`'s docstring carries the corrected numbers plus a practical caveat for callers: correction quality depends on `rho_at_scales` being a low-noise (large-enough-K) estimate to begin with -- an inherent property of Richardson-style extrapolation's noise amplification, not a bug in this function.
  - Also tested: whether more noise-scale points (4, 5, or denser spacing) improve *exact* interpolation further. They don't -- more points make Lagrange coefficients larger (worse with denser spacing, a Runge's-phenomenon-like effect), amplifying noise further: mean delta drops from +0.148 (3 points, the original choice) to +0.081 (5 points, same spacing) to -0.220 (5 points, denser spacing -- actively worse than not correcting at all on some channels).
- **Added**: `dense_evolution.mitigation.polynomial_extrapolate` -- least-squares polynomial extrapolation to zero noise, generalizing `richardson_extrapolate`. At exactly `degree + 1` points it's mathematically identical to exact interpolation (verified to 1e-12); with MORE points it becomes an overdetermined fit that trades a little bias for real variance reduction instead of the exact interpolation instability above. `zne_density_matrix` now uses this (degree=2 default) instead of `richardson_extrapolate` -- **at the standard 3-point setup this changes nothing** (mathematically identical, all prior tests pass unchanged), but makes "pass more noise-scale points" safe instead of a trap.
  - **Real, measured gain, fixed total measurement budget** (`experiments/matrix_healing_fixed_budget.py`, the fair comparison -- splitting the same total number of Monte Carlo trajectories differently, not spending more): 3 points x K=400 (1200 total, classic 3-point ZNE) vs. 5 points x K=240 (1200 total, degree=2 fit) vs. 7 points x K=171 (~1200 total). 5 points matches or slightly beats the 3-point mean improvement (+0.150 vs +0.148) with **19% lower variance** (std 0.050 vs 0.062) -- a free reliability gain at equal experimental cost. 7 points trades a little mean (+0.132) for 30% lower variance (std 0.043), a genuine tradeoff point.
  - **Tested and rejected, for honesty**: (1) applying `zero_noise_extrapolation`'s existing predictive-healing coefficient perturbation (`calculate_delta_preemp`, originally designed for scalar coherence signals around target_sigma_ideal=10) to density matrices, using either matrix purity or Jensen-Shannon divergence (`dense_evolution.mps._jsd_vectors`) as the coherence signal -- no measurable effect even amplified 100x over its default strength. (2) Selecting the polynomial degree per-run via leave-one-out cross-validation on the noisy points themselves -- worse than a fixed degree (mean +0.113 vs +0.150, higher variance), because LOOCV on few noisy points is biased toward under-fitting. (3) Selecting degree via the same JSD coherence signal -- no exploitable correlation found between JSD and which degree actually performs best (degree 2 and 3 win about equally often across the whole measured JSD range). None of these are shipped; `degree=2` fixed remains the best-tested default.

### v8.1.37
- **Added**: `dense_evolution.mitigation.zne_density_matrix`/`project_to_physical`/`uhlmann_fidelity` -- Zero-Noise Extrapolation extended from scalars/vectors to full density matrices, for noisy simulations tracked as ρ rather than a single statevector. `project_to_physical` implements Smolin, Gambetta & Smith's "Maximum Likelihood, Minimum Effort" (2012, arXiv:1106.5458) eigenvalue-projection algorithm (transcribed and checked against the paper's own worked numeric example before use) to correct the raw `richardson_extrapolate` output on a matrix stack -- polynomial extrapolation across noise-scaled density matrices is not itself guaranteed to be a valid (positive-semidefinite) density matrix, even when every input was. `uhlmann_fidelity` is a validation-only utility (`F(ρ_A,ρ_B) = (Tr√(√ρ_A ρ_B √ρ_A))²`, reduces to `|⟨ψ_A|ψ_B⟩|²` for pure states, verified directly) -- deliberately not accepted as an input anywhere in the correction path, keeping "the ideal state is a grading criterion, not something the algorithm gets to see" structural rather than a convention callers have to remember on their own (the general principle: an error-mitigation technique that needs to know the answer already isn't one).
  - **Honest, measured result** (`experiments/matrix_healing_zne.py`, reproducible): 2-qubit Bell state, `NoiseModel` depolarizing noise at base_p=0.05, scales 1x/2x/3x, a 200-trajectory Monte Carlo density-matrix estimate per scale, averaged over 4 independent random seeds -- raw noisy fidelity ~0.865, corrected (extrapolated + projected) fidelity ~0.947, a real improvement of ~+0.08, positive individually on every seed tested (range +0.060 to +0.106). This is one measured data point in one noise regime, not a general guarantee -- other noise models, circuits, or noise strengths are untested and may behave differently; the docstrings say exactly this, not more.
  - This formalizes an experiment built directly on top of v8.1.36's `richardson_extrapolate` complex-dtype fix -- the same experiment run *before* that fix gave an invalid, misleadingly negative result (corrected fidelity *worse* than raw) purely because the bug was silently discarding the density matrices' imaginary parts, not because the technique doesn't work.

### v8.1.36
- **Fixed**: `dense_evolution.mitigation.richardson_extrapolate` (and `zero_noise_extrapolation`'s healing-adapted branch) hardcoded `dtype=jnp.float64` for `expectation_values`, silently discarding the imaginary part of complex input with only a low-signal `ComplexWarning`, no explicit error. Real inputs (the only case exercised by any existing caller) were unaffected, but any complex-valued caller -- e.g. extrapolating density-matrix entries, which are complex off-diagonal in general -- got a silently wrong, purely-real result. Found while building a density-matrix-healing experiment on top of this function; verified directly pre-fix (`richardson_extrapolate([1+2j, 3+4j], ...)` returned a purely real value, dropping real information). Fixed by picking `values`'s dtype from the input itself (`jnp.complex128` if `np.iscomplexobj(expectation_values)`, else `jnp.float64` -- identical behavior to before for every existing real-valued call site, verified: all prior tests pass unchanged).

### v8.1.35
- **Added**: `MPSSimulator.run_circuit_jit(ops)` -- a `jax.lax.scan`-fused, `@jax.jit`-compiled whole-circuit execution path for `MPSSimulator`, which previously had zero `@jax.jit` anywhere in the file: every gate ran as an eager JAX call, and the adaptive bond-dimension search inside `_svd_truncate` was a Python `while` loop forcing a host-device sync on every 2-qubit gate. Measured directly on a 60-qubit/428-gate stress circuit (the same one used to originally diagnose this): **88.9s (eager) -> 0.74s (fused, steady-state post-compile)**, ~120x faster, now within range of Qiskit Aer's MPS backend (0.64s) on the same circuit -- same physics exactly (`chi_used=16`, `budget_violations=0`, `avg_JSD=0.0000`, matching the eager run's own summary).
  - The two root causes, quantified before fixing: (1) zero JIT fusion -- confirmed via direct grep, not assumed; (2) non-adjacent 2-qubit gates expand into a SWAP chain at runtime, multiplying the real SVD-bearing gate count 2.4x on the stress circuit (187 logical 2-qubit gates -> 451 real `apply_gate_2q` calls, counted directly).
  - New technique (no precedent for this specific problem in `chunk.py`'s own earlier JIT fusion, issue #5, whose geometry never changes gate-to-gate): the bond-dimension search is now a single vectorized pass (JSD computed for every candidate truncation size at once via `jax.vmap`, replacing the incrementing `while` loop) over gamma/lambda tensors padded to a *fixed* `max_bond` size always, with the real bond dimension tracked as a traced scalar and everything beyond it zero-masked rather than dynamically sliced (`jax.lax.scan`'s carry requires constant shape/dtype every step). Non-adjacent gates are expanded into their SWAP-chain-equivalent adjacent-only op sequence at Python pre-compile time instead of via runtime dispatch, mirroring `chunk.py::_compile_multi_chunk_ops`'s "all gate-identity branching happens before tracing starts" principle.
  - `apply_gate_1q`/`apply_gate_2q`/`_apply_nonlocal_2q` (the eager path) are unchanged and still available -- `run_circuit_jit` is a new, additional entry point, not a replacement. Explicit, intentional trade-off: the fused path pads every gamma/lambda to `max_bond` for the rest of the instance's lifetime after the call, trading this module's adaptive-memory benefit (its whole point for very large, low-entanglement circuits) for speed -- use the eager methods directly when memory, not speed, matters more.
  - Verified in 5 separately-checked stages, each against real data before moving to the next: (1) the vectorized bond-dimension search matches the eager `while` loop's `(chi_new, jsd_val)` exactly across 171 real cases from real circuits, including the budget-violation fallback; (2) the SWAP-chain pre-expansion matches the real runtime dispatch's exact call sequence across all 330 `(q1, q2)` pairs for 3 qubit counts, plus an end-to-end final-state replay check; (3) the fused kernel matches `DenseSVSimulator` on real entangling circuits (TVD < 1e-6) and matches the eager path bit-for-bit (fidelity 1.0 to machine precision) even under a genuine `budget_violations` truncation scenario, both dtypes; (4) `entanglement_entropy`/`_bond_history`/`jsd_per_bond`/`truncation_errors`/`budget_violations` end up populated identically to the eager path (machine-precision match), sourced from `jax.lax.scan`'s stacked per-step diagnostics instead of Python list appends; (5) the real speed number above.
  - `contract_to_statevector` generalized (`.squeeze(axis=0)`/`.squeeze(axis=-1)` -> `[0]`-indexing) to work correctly on both the original unpadded boundary tensors (identical behavior, verified against the full existing test suite) and the new padded ones.

### v8.1.34
- **Fixed**: `QASMParser._eval_param` silently returned `0.0` for *any* gate-parameter expression it couldn't evaluate -- a malformed expression like `rx(pi * / 2) q[0];` parsed successfully and silently produced `rx(0.0)`, a different, valid circuit, with no signal a typo had happened. Found via an independent code-review report, reproduced directly (`ast.parse('pi * / 2', mode='eval')` itself correctly raises `SyntaxError` -- the bug was a blanket `except Exception: return 0.0` around it, added when the previous raw-`eval()` code-execution vulnerability was fixed, that swallowed genuine syntax errors along with rejected/malicious expressions). Now raises `ValueError` instead, for both cases -- malformed expressions and disallowed/malicious ones alike (still structurally blocked by the same AST node-type whitelist either way, just explicit about it instead of silent). Same class of silent-wrong-behavior issue already fixed for unknown gate names and mismatched parameter batches (v8.1.27, issues #4/#6). 2 existing security tests (`test_eval_param_blocks_sandbox_escapes`, the full-parse exploit test) updated to expect the raised exception instead of a silent `0.0`; 2 new regression tests added for the originally-reported malformed-expression case.

### v8.1.33
- **Fixed**: `MPSSimulator._apply_nonlocal_2q` silently swapped which qubit acted as gate-argument-1 vs gate-argument-2 whenever a non-adjacent 2-qubit gate (`abs(q1-q2) != 1`) was called with `q1 > q2` -- e.g. `apply_cx(ctrl=3, tgt=1)` -- because the SWAP chain always ends up applying the gate at `(min(q1,q2), min(q1,q2)+1)` regardless of the caller's original argument order. For an asymmetric gate like CNOT this silently applied the wrong gate (control/target inverted) with no error or warning. Found via an independent code-review report, reproduced directly before trusting it: building a 4-qubit GHZ state via `H(0), CX(0,3), CX(3,1), CX(1,2)` gave fidelity 0.25 against the exact GHZ state (`(|0000⟩+|1001⟩)/√2` instead of `(|0000⟩+|1111⟩)/√2`). Fixed with the same q1>q2 normalization `apply_gate_2q`'s adjacent-qubit branch already used (transpose the gate's tensor axes, swap q1/q2) applied before the SWAP chain runs -- same fidelity check now reads 0.9999999657714559. Affects `apply_cx`/`apply_cz`/`apply_swap`/`apply_ccx` for any non-adjacent qubit pair called in decreasing order, reachable from `dashboard_core`'s `engine='mps'` option on any QASM circuit with such a gate. 6 new regression tests in `test_mps.py` (direct repro, cross-check against `DenseSVSimulator`, non-adjacent unordered-control Toffoli, both float32/float64).
- **Checked, not changed**: the same external report also flagged `MPSSimulator.get_top_k_probable_states`'s amplitude summation as a potential bug -- independently re-verified against the exact contraction (matches to machine precision) and against the module's own documented history of already fixing exactly this class of bug. No live issue found. Its performance observation (the `while` loop in `_svd_truncate` isn't `jax.jit`-compiled) is real but is a speed question, not correctness, and is deliberately out of scope here.
- **Added**: `dashboard_core.run_md_telemetry` now computes real molecular-dynamics-style telemetry (exact phase evolution under the selected diagonal Hamiltonian, plus `dense_evolution`'s own `NoiseModel` driving a physically-grounded thermal-noise channel) whenever a compatible Hamiltonian and the circuit's current statevector are supplied, instead of always returning the synthetic placeholder data (`run_md_simulation_dummy`, documented as a placeholder since it was first written) that shipped every prior version. Falls back to the same labeled mock when no compatible Hamiltonian is active -- the result now carries `df.attrs['is_real']`/`df.attrs['note']` so callers (and the UI) can show honestly whether a given MD panel is real or a demonstration, rather than presenting synthetic numbers as if they were physics. Energy/entropy/purity are all genuinely computed (purity via an ensemble-averaged density matrix, exact given the ensemble -- every `LIBRERIA_HAMILTONIANE` entry is ≤6 qubits, so this stays cheap). Wired into both `ui_pages/quantum_simulator.py` (Streamlit) and the new interactive panel below with a visible "DATI REALI" / "MOCK" badge on the MD tab.
- **Added**: `dashboard_core.launch_interactive_panel` -- a full ipywidgets-based interactive dashboard for Colab/Jupyter, built directly on `dashboard_core`'s existing functions (not a port of the old `legacy/dash.py` notebook export). Same panels as the Streamlit page (Overview, Fisica Stato, Mosaico, VQE Results, MD Results, Performance, 3D Helix, Hamiltonian, Mitigation (ZNE)) and the same sidebar controls (circuit source, engine, noise, ZNE/predictive healing, VQE, custom Hamiltonian, MD), driven by ipywidgets instead of Streamlit's rerun model -- stays inside a single notebook cell's output, no external tunnel/link. Requires the `dashboard` extra's `ipywidgets>=8.0.0` (now included in that extra).

### v8.1.32
- **Added**: `dashboard_core.mitigation_runner.run_mitigation_sweep`/`dashboard_core.mitigation_panel.build_panel_mitigation` -- Zero-Noise Extrapolation and predictive healing, wired into the Streamlit dashboard as a real feature for the first time (`dense_evolution.mitigation`/`dense_evolution.healing` existed in the core package but had zero references anywhere in the dashboard before this). Runs the active circuit at 3 noise scales (`1x/2x/3x` the sidebar's noise probability) and extrapolates to zero noise via `zero_noise_extrapolation`, reused exactly as-is -- no reimplementation of the Richardson/healing math in the dashboard layer. New sidebar section ("🩹 Error Mitigation (ZNE)") and a new "Mitigation (ZNE)" tab showing fidelity-vs-noise-scale and ideal/raw/ZNE-corrected probability overlays; the healing-adapted path's coherence signal (`sigma_at_base_noise`) reuses the shot-noise binomial sigma already computed and shown elsewhere in the dashboard (`build_panel_overview`'s NISQ Shot Histogram) as a pragmatic proxy, documented as such rather than presented as a first-principles derivation.
- **Fixed**: `dense_evolution.mitigation.richardson_extrapolate` raised `ValueError: Incompatible shapes for broadcasting` whenever `expectation_values[i]` was itself array-valued (e.g. a full probability distribution per noise scale, not a bare scalar) -- found building the panel above, where each noise scale's "expectation value" is naturally a whole probability vector. Root cause: stacking `expectation_values` via `jnp.asarray` and multiplying by the `(n,)`-shaped Lagrange coefficients relied on JAX's default trailing-axis broadcast alignment, which pairs the coefficients against the *last* axis of the stacked array instead of the *leading* "one row per noise scale" axis. Fixed by reshaping the coefficients to broadcast against the leading axis explicitly and summing over `axis=0` -- a no-op reshape for the pre-existing scalar case (verified: all prior scalar-input tests pass unchanged), now also correct for vector/array-valued inputs.
- **Fixed**: `dashboard_core.run_simulation`'s `seed` parameter only ever reached `np.random.seed()` (shot sampling) -- for a JAX statevector (the normal case), the actual noise channel applied via `NoiseModel.apply_to_sv` was never given `rng=`/`jax_key=`, so it silently fell back to OS-entropy randomness regardless of `seed`. Two calls with the identical seed produced different noisy results. Found as a flaky test (`run_mitigation_sweep`'s 3-noise-scale sweep, same seed passed to each scale, expected -- and needed, for a meaningful extrapolation -- the same underlying noise realization scaled by `p`, got a fresh random one each time). Fixed by passing `rng=np.random.default_rng(seed)` explicitly to `apply_to_sv`.
- **Refactor**: `dashboard_core.py` (1900-line single file) split into a package (`dashboard_core/`: `qasm_library`, `hamiltonians`, `simulation_runner`, `vqe_engine`, `md_telemetry`, `plot_theme`, `metrics`, `panels`, `helix_3d`, `mitigation_runner`, `mitigation_panel`, `provenance`) behind a backward-compatible `__init__.py` façade re-exporting the same public surface -- `import dashboard_core as dc` and every `dc.xxx` call site unchanged, verified via the existing test suite passing identically with zero test-file edits. `_series_plot`/`_series_enhanced`/`_energy_enhanced` (three near-duplicate time-series plotting functions, `_energy_enhanced` duplicating `_series_enhanced`'s logic inline instead of calling it) unified into one parameterized `_series_plot`. Hardcoded color literals in `build_panel_fisica`/`vqe_results`/`md_results`/`performance` that exactly matched an existing `plot_theme.C` value now reference it instead of repeating the literal -- zero visual change, by construction; near-miss shades (close but not identical to a `C[...]` value) deliberately left alone rather than forced onto the "closest" key, which would have been a small but real, unauthorized color drift.
- **Also fixed**: `.github/workflows/ci.yml` never ran `test_mitigation.py` or `test_mps.py`.
- **Refactor (`ui_pages/`, repo-only, not part of the pip package)**: the gradient header banner, the "run → session_state → guard → render" early-return pattern, and the neutral AI-shield metadata dict were each copy-pasted across 2-3 of `quantum_simulator.py`/`vector_healing.py`/`quantum_scars.py`/`ai_middleware.py` instead of shared. New `ui_pages/components.py` helpers (`render_page_banner`, `render_run_guard`, `render_matplotlib_figure`) and `AI_SHIELD_NEUTRAL_META` replace them -- same visuals, one place to change instead of three. Fixes, as a side effect, three independently-drifted hardcoded version numbers in the page banners (`v8.1.9`/`v8.1.7`/`v8.1.22`) plus a fourth in `app_dashboard.py`'s own `st.set_page_config`/docstring -- all four now read `dense_evolution.__version__` instead of a stale literal. Verified: existing `test_ai_middleware.py`/`test_quantum_scars.py` pass unchanged (22/22); all `ui_pages` modules import cleanly; the actual Streamlit server started and served all 3 page routes with no server-side exceptions in the log (no automated UI test exists for this layer, per the pre-existing documented gap -- this is the closest verification available without a browser screenshot tool in this environment).

### v8.1.31
- **Changed**: `dashboard_core` (the compute/panel layer behind the Streamlit dashboard) is now part of the installable package (`pyproject.toml`'s `packages`), not repo-only -- `pip install dense-evolution[dashboard,jax]` alone now gives `import dashboard_core as dc` with the full 53-circuit QASM library and every panel builder, no `git clone` required. Superseded within the same release by v8.1.32's package split (`dashboard_core.py` → `dashboard_core/`) -- the packaging mechanism changed from a single `py-modules` entry to a real package directory, but the end result (`pip install` alone is sufficient) is unchanged.

### v8.1.30
- **Added**: `Chunk.run_chunk_distributed` -- dispatches the multi-chunk kernel across a real JAX device mesh (`jax.shard_map` + `jax.lax.ppermute`) instead of one process's RAM, one physical chunk per device (issue #1, v1 scope: `jax.device_count() >= num_chunks`). Cross-chunk gate mixing becomes point-to-point communication via `ppermute`, keyed on the fixed XOR-stride pairing between chunk indices -- the same pairwise-exchange pattern real distributed statevector simulators use; gates entirely local to a chunk, or with a chunk-select control and a local target (decidable from a device's own chunk index alone), need no communication at all. Verified for correctness against `DenseSVSimulator` on simulated multi-device CPU (`XLA_FLAGS=--xla_force_host_platform_device_count=N`) across all 6 gate/qubit-location cases individually, randomized mixed circuits at `num_chunks` in {4, 8}, and both dtypes -- CI now runs these in a dedicated step with 8 simulated devices (they skip cleanly on a single-device run). Real GPU-cluster network performance is not and cannot be measured in a single-device CI environment -- honest scope: this validates the sharding/communication logic is correct, not real multi-GPU throughput.
- **Fixed while building it**: the `ppermute` permutation for each chunk-select qubit was built inside a Python list comprehension with the loop variable referenced directly in the lambda body -- classic late-binding closure bug, every branch silently used the *last* qubit's stride regardless of which was actually selected at runtime (only the branch for the last-iterated qubit happened to come out correct, which is exactly the failure pattern that surfaced it: `q0`/`q1` wrong, `q2` right, on a 3-chunk-select-qubit circuit). Fixed by binding the fully-precomputed static `perm` list as a default argument, evaluated eagerly at lambda-creation time instead of looked up by reference at call time.
- **Also fixed**: `.github/workflows/ci.yml` never ran `test_mitigation.py` or `test_mps.py` -- both existed and passed locally but had never actually executed in CI. Added to the main test run.

### v8.1.29
- **Added**: `dense_evolution.NoiseSpec` -- a JAX PyTree (`model`/`qubits` static, `p`/`jax_key` as leaves, registered via `jax.tree_util.register_pytree_node`) that `circuit_to_energy_fn`'s `energy_fn` now accepts as a fourth `noise=` argument, applying `NoiseModel.apply_to_sv` natively inside the same traced computation as `theta` (issue #8, scoped up from #7's narrower fix). Removes the need for an external, Python-side noise-application step and manual PRNG key bookkeeping around a training loop -- `jax_key` is a pytree leaf, so it flows through `jax.jit`/`jax.grad`/`jax.vmap`/`jax.lax.scan` the same way any other JAX array does, with no OS-entropy fallback (reproducibility is structural, not opt-in). Verified: reproducible from the same key, energy differs from the ideal circuit under nonzero `p`, gradient w.r.t. `theta` flows correctly through the noisy pipeline, `jax.vmap` over a batch of independent keys works with no external Python loop, `jax.jit`-wrapped result matches eager evaluation exactly.
- **Fixed**: found while wiring the above -- `NoiseModel.apply_to_sv`'s `if model == 'ideal' or p <= 0.0: return sv` early-exit raised `TracerBoolConversionError` as soon as `p` became a traced value (e.g. a `NoiseSpec` leaf under `jax.jit`) -- exactly the case #8 asked to enable. Fixed with a targeted `try/except` around only the `p <= 0.0` optimization: every channel's math already reduces to a no-op at `p=0` (`fire = r < p` is always `False`), so skipping the shortcut and falling through is still correct, it just loses the eager-mode fast path when `p` can't be checked concretely.
- **Note**: `noise.p` is technically a differentiable pytree leaf, but the gradient through it is ~0 almost everywhere in practice -- the existing channels sample via a hard threshold (`fire = r < p`), not usefully differentiable without a smooth relaxation (e.g. Gumbel-softmax). Not addressed here, out of scope for what #8 asked (removing the state/key workaround, not making noise strength itself gradient-optimizable).

### v8.1.28
- **Note**: v8.1.27 was published to PyPI from a stale local checkout that predated the `mitigation.py` module and the #4/#6/#7 fixes below -- the installable v8.1.27 package on PyPI does **not** contain them, despite this changelog. PyPI doesn't allow re-uploading files under an already-published version, so this release exists specifically to ship the real content. If you're on 8.1.27, upgrade to 8.1.28 -- don't rely on 8.1.27's changelog matching what you actually have installed (same situation as v8.1.15, see that entry below).

### v8.1.27
- **Added**: `dense_evolution.mitigation` -- a Zero-Noise Extrapolation orchestrator (`richardson_extrapolate`, `zero_noise_extrapolation`, both exported from package root) using the field's standard ZNE vocabulary, composed on top of `healing.py`'s existing primitives rather than duplicating or renaming them. Motivated by the ZNE+predictive-healing combination previously existing only as hand-written logic (`_static_richardson`/`_adaptive_healing_richardson`) inside a downstream repo's test file, calling `calculate_delta_preemp` as one ingredient with no reusable, standard-named entry point in the package itself -- undiscoverable by name for anyone (human or AI) reading `dense_evolution` looking for "ZNE". `richardson_extrapolate` generalizes to N arbitrary noise factors (verified: reduces exactly to the textbook `(3, -3, 1)` 3-point coefficients at `noise_factors=(1,2,3)`, and is exact on synthetic linear data); the healing-adapted path is scoped to exactly 3 noise factors -- the only case it has been derived and verified against (cross-checked numerically against the original 3-point formula) -- and raises `NotImplementedError` rather than silently generalizing an unverified formula for other point counts.
- **Breaking**: `run_circuit`/`run_circuit_jit_beast_mode`/`run_parametric_batch_jit` now raise `ValueError` on an unrecognized gate name instead of silently dropping the gate (issue #4). Verified pre-fix: `run_circuit_jit_beast_mode([('h', 0), ('ch', 0, 1), ('x', 2)])` executed only `h`/`x`, `ch` vanished with no error or warning -- a typo in a gate name (`'crx'` instead of `'crz'`) used to silently run a different circuit than the one written. Anyone relying on the old silent-skip behavior will now get an explicit error instead -- there is no legitimate use case for a gate name silently doing nothing.
- **Breaking**: `run_parametric_batch_jit` now validates `parameter_batch.shape[1]` against the actual number of parametric gates (rx/ry/rz/p/u1/phase/cp/crz/cphase) in `base_circuit`, raising `ValueError` on a mismatch instead of letting JAX's default out-of-bounds indexing clip silently (issue #6). This is the same positional-slot contract responsible for two independently-discovered corrupted-results bugs in a downstream repo (`Dense-Evolution-Ising-Tests` v2.1.0): a literal-float rotation gate still consumes a column, it is never exempt, and a grid built assuming otherwise used to produce a plausible-looking but wrong statevector with no signal that anything was off.
- **Fixed**: `NoiseModel.apply_to_sv` used to silently ignore `rng` whenever `sv` was a JAX array, drawing from OS entropy instead -- a caller seeding `rng = np.random.default_rng(42)` for reproducibility got a different, non-reproducible result on every call, silently (issue #7). Verified pre-fix directly: two consecutive calls with the identical seeded `rng` and identical input diverged. Found while reviewing an end-to-end QML+noise+ZNE test script combining `circuit_to_energy_fn`, `NoiseModel`, and the new `zero_noise_extrapolation` under one `jax.value_and_grad` -- gradients did flow correctly through the whole pipeline (that part worked as intended), but the epoch-to-epoch loss instability in that script traced back to this, not to the optimizer or to `mitigation.py`. Fixed: when `sv` is a JAX array and `jax_key` isn't given explicitly, `rng` (if given) now *derives* the JAX key (`rng.integers(...)` seeds `jax.random.PRNGKey`) instead of being ignored -- a fresh, identically-seeded `rng` now reproduces the exact same sequence of keys across separate runs, matching the guarantee the NumPy path already gave. `jax_key`, when given explicitly, still takes precedence over a derived one. Making `apply_to_sv` itself composable inside a single `jax.jit` block (no internal Python-level branching or OS-entropy side effects) is a larger, separate redesign -- not done here, see issue #7 for the remaining scope.

### v8.1.26
- **Verified, no bug found**: two areas flagged as untested in a prior RAM-constrained environment -- `Chunk`'s genuine multi-chunk dispatch (`num_chunks > 1`) on a 2-qubit gate spanning the very first (chunk-select) and very last (local) qubit of the register, and `QASMParser`'s handling of `for`-loops with an unresolvable bound (an undeclared bound variable, so `_resolve_int_expr` returns `None`) -- both confirmed correct. Closed the real test-coverage gap: 4 new tests in `TestChunkMultiPiece` (the long-range 2-qubit gate case, plus `MemoryPressureError` actually firing on simulated low RAM for both the `num_chunks==1` and `num_chunks>1` code paths -- previously only ever exercised indirectly, never asserted on directly) and 6 new tests in `TestQASMForLoop` (unresolvable `for` bound stripped cleanly with following code preserved -- verified via probability comparison, not just the op list -- `while` blocks, multiple unresolvable constructs in sequence, and a resolvable `for`-unroll immediately followed by an unresolvable `if`-strip).

### v8.1.25
- **Fixed**: `MPSSimulator._svd_truncate` used to silently violate `jsd_budget` whenever `max_bond` was too small for the circuit's real entanglement -- the `while jsd_val > self.jsd_budget and chi_new < max_possible` loop exits with no signal once `chi_new` hits `max_bond`, even if `jsd_val` is still far above budget. `summary()`'s `avg_JSD` only reports the mean of the per-step local errors, not the accumulated global error, so it can read deceptively low while the final contracted state is badly wrong -- verified directly on an 8-qubit/15-layer entangling circuit with `max_bond=2`: TVD ~0.97 against `DenseSVSimulator` (a near-total mismatch) while `avg_JSD` read a reassuring-looking 0.0534. Added a `budget_violations` counter (incremented every time this happens, exposed in `summary()`) and a `UserWarning` on the first violation (`"bond dimension capped at max_bond=..., jsd_budget=... not honored ... results may be unreliable"`) -- not an exception, so existing code that tolerates the tradeoff keeps working, but now with an explicit, checkable signal instead of a silently-optimistic average.

### v8.1.24
- **Fixed**: `dense_evolution.healing.calculate_phi_ab` raised `ValueError: Clip received a complex value` when called with complex statevectors (e.g. `sim.get_statevector()`) -- `jnp.dot(semantic_change, ipg_vector)` is the bilinear (non-conjugated) product on complex arrays and stays complex, which then hit `jnp.clip` at the end of the function. Fixed by using `jnp.real(jnp.vdot(...))` -- the correct Hermitian-inner-product real part, identical to `jnp.dot` for the real-valued inputs every existing caller already uses (`ia_utils.vector_healing.enhanced_dense_healing_hybrid`), and now also correct for genuinely complex input. Verified against a manual NumPy `Re(vdot(...))` calculation on a case with nonzero imaginary amplitudes (H+S gates), not just the crash repro (H+CX alone never produces a nonzero imaginary part, so it only proved "doesn't crash," not "computes the right number").
- **Fixed**: `ia_utils.vector_healing.median_healing`/`enhanced_dense_healing_hybrid` emitted `RuntimeWarning: Mean of empty slice` whenever an input column was entirely `NaN` -- `np.nanmean` on an all-NaN slice returns `NaN` with a warning (silently caught and zeroed by the very next line, so the *output* was already correct, only the warning was noise). Fixed by pre-replacing whole all-NaN columns with `0.0` before calling `nanmean`, so it never sees an empty slice -- verified byte-identical output, zero warnings, in both functions (the preprocessing block was duplicated verbatim in each).
- **Fixed**: `enhanced_dense_healing_hybrid`'s `fallback_triggered` metadata flag used to reflect only the internal Phi-Trigger heuristic classifying a step as "static", regardless of whether the input actually contained any `NaN`/`Inf` -- verified directly that a clean, uncorrupted random Gaussian array (no corruption at all) still came back `fallback_triggered=True`. Root cause: the heuristic is tuned for trajectories with real underlying dynamics (verified separately against a real noisy VQE run, where it correctly stayed quiet at low noise and fired at high noise) and mistakes pure structureless IID noise -- which has no coherent trend to recognize as "genuine change" -- for anomalous static behavior on nearly every step. `fallback_triggered` is now gated on the original input actually containing `NaN`/`Inf` (checked before sanitization) in addition to the fallback having fired -- the dashboard's "Fallback scattato" indicator will now only light up for genuine NaN/Inf corruption, not general noise-driven Phi-Trigger corrections (those corrections still happen internally, they're just no longer mislabeled as a NaN/Inf fallback).
- **Docs**: removed two README rows (`kappa_stabilization`, `richardson_integration`) describing functions that never existed in `dense_evolution/healing.py` (confirmed against the full 140-line file -- 7 functions plus `MemoryReflectionEngine`, neither name present anywhere), and corrected two code examples that still showed a `median_fallback_threshold` parameter on `enhanced_dense_healing_hybrid` that was removed from the actual function signature back when its corresponding UI slider was dropped (commit `69fc8a4`) without updating the docs to match.

### v8.1.23
- **Added**: `MPSSimulator` (`dense_evolution/mps.py`, re-exported from the package root) -- a JAX-backed Matrix Product State simulator with adaptive SVD-truncated bond dimension. Verified exact against `DenseSVSimulator` on entangling circuits (TVD=0). Selectable as a dashboard engine (Quantum Simulator page) for circuits up to 24 qubits; for larger low-entanglement circuits, `get_probabilities_sampled`/`get_top_k_probable_states` scale to hundreds of qubits without ever materializing a `(2**n,)` statevector.
- **Fixed**: large-qubit circuits submitted through the dashboard used to crash the whole Streamlit process (uncatchable OS-level OOM) instead of failing cleanly. They now route automatically through the existing `Chunk` anti-OOM wrapper, which raises a catchable, informative `MemoryPressureError` instead.
- **Changed (breaking)**: JAX is now a **required core dependency** (`dependencies`, not `optional-dependencies`) -- `pip install dense-evolution` installs it by default, no `[jax]` extra needed anymore. Every simulator backend in this package already required JAX in practice; the previous `try/except ImportError` numpy-fallback path was never a real, maintained alternative and is no longer reachable (the numpy code itself is left in place for reference, just dead). If you were pinning an environment without JAX and relying on degraded-numpy behavior, this release will break that -- install `dense-evolution<8.1.23` to keep the old behavior.

### v8.1.22
- **Added**: `ui_pages/quantum_scars.py` — a new "Quantum Scars" dashboard page (`app_dashboard.py`'s `st.navigation`), an interactive live demo of the PXP quantum many-body scar model (Rydberg blockade): builds `H_PXP` via sparse Pauli operators, exact-diagonalizes it (`scipy.linalg.eigh`, cached via `st.cache_resource` keyed on qubit count — the first use of that decorator in this codebase, since diagonalizing a dense `2**n_qubits` matrix is genuinely expensive and depends only on that one slider), propagates a Néel initial state under real-time Hamiltonian evolution, injects real noise via `NoiseModel.apply_to_sv`, and lets you compare fidelity revival with no protection, a cheap constraint-subspace projection (no extra diagonalization needed — a combinatorial mask of which computational-basis bitstrings have no two adjacent 1-bits), or an idealized exact-eigenstate "tower" projection. Distills the investigation already published at [`quantum_scar_investigation`](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/tree/main/scripts/quantum_scar_investigation) — which found no genuine scar in Dense Evolution's own frustrated Ising grids (wrong observable + gauge equivalence), then validated the same verification pipeline against PXP, where scars are real and well documented — into something runnable instead of only readable. Verified via `streamlit.testing.v1.AppTest` (real button-click simulation, zero exceptions) and a new `test_quantum_scars.py` (17 tests, all passing) unit-testing the numerical core directly: validity-mask combinatorics, `H_PXP` Hermiticity and Hilbert dimension, fidelity=1/norm-preservation under propagation, and both projections staying normalized with zero weight outside their target subspace.

### v8.1.21
- **Added**: `QASMCircuit.__iter__` — duck-types a parsed circuit as an iterable of the same tuples `.to_tuples()` returns, so it works anywhere a plain circuit list is expected (`Chunk.run_chunk`, `QuantumTranspiler.transpile`, ...) without remembering to call `.to_tuples()` first. Found via a user's own Colab testing: `Chunk.run_chunk(QASMParser().parse(qasm))` — a very natural thing to try — raised `TypeError: 'QASMCircuit' object is not iterable`.
- **Fixed / Performance**: `Chunk`'s multi-chunk dispatch (`num_chunks > 1`) used to apply every gate through a Python loop calling non-JIT `apply_gate_1q`/`apply_gate_2q` — measured 6x slower than `run_circuit_jit_beast_mode` on an identical workload. Replaced with a single `jax.lax.scan` over the whole circuit operating directly on the stacked `(num_chunks, chunk_dim)` representation — never materializing a `(2**n_qubits,)` array, preserving the anti-OOM property `Chunk` exists for. The 6 gate/qubit-location cases were ported formula-for-formula from the old Python-loop implementation and verified case-by-case against it (all 18 pre-existing `TestChunkMultiPiece` tests, which cross-check against `DenseSVSimulator`, pass unchanged against the new kernel) before the old code was removed. Gate coverage is now built via `GATE_IDS` instead of the old `GATES`/`PARAMETRIC_GATES` lookup, aligning it with beast-mode's own coverage. Measured speedup on the exact benchmark that surfaced the problem (10 qubits/200 gates/4 forced chunks): **2.2s → 0.42s**, now close to beast-mode's 0.37s on the same non-chunked workload.
- **Note**: a Colab report of "17s vs milliseconds" that prompted this investigation turned out, on reproduction, to be `num_chunks==1` (not the multi-chunk path at all) — 0.49s locally on the identical circuit, most likely first-time JIT compilation overhead on Colab's specific hardware rather than a code defect. The 6x slowdown that *was* real and is fixed here was found and confirmed with a separate synthetic benchmark (`num_chunks` forced via monkeypatch), not the original report.

### v8.1.20
- **Fixed**: `from_pennylane`/`run_pennylane_circuit` silently renumbered qubits whenever wires weren't touched in ascending order — both PennyLane's `qml.to_openqasm` and the older `tape.to_openqasm()` number exported QASM qubits by first-touch order, not by actual wire index (e.g. `PauliX(wires=2)` then `CNOT(wires=[2,1])` exported as `x q[0]; cx q[0],q[1];`, silently mapping wire 2→q[0] and wire 1→q[1]). Found via independent fuzz testing (20 random circuits touching 4 wires in random order: 9/20 matched PennyLane's own results before the fix, 20/20 after). Fixed by passing an explicit `wires=` argument — the device's declared wire order for a QNode, the tape's own wires sorted ascending for a bare tape — forcing the true wire order into the export instead of relying on touch order.
- **Fixed**: `NoiseModel.apply_to_sv`'s `depolarizing` channel (and the `combined` channel's depolarizing sub-channel) picked which Pauli error (X/Y/Z) to apply using thresholds `p/3` and `2p/3` compared against a draw uniform on the *full* `[0,1)` range — but that draw should only ever decide *which* Pauli fires, independent of the overall fire-rate `p`, so the thresholds needed to be the fixed values `1/3` and `2/3` instead. The bug skewed every depolarizing/combined-noise circuit heavily toward Z regardless of `p` (verified: at `p=0.3`, `P(X|fire)=P(Y|fire)=10%`, `P(Z|fire)=80%` instead of the documented 33.3% each — confirmed both via an isolated 100k-sample trace of the raw branch logic and via full statevector simulation, both matching the bug's predicted skew to within statistical noise). Found via independent statistical fuzz testing comparing measured frequencies against the analytic prediction — a test that only checks "the channel runs without crashing" would never have caught this. `bitflip`, `phaseflip`, and `amplitude_damping` were verified unaffected (correct by construction, don't use this three-way branch).
- **Docs**: opened a tracking issue for a related robustness gap found during the same fuzzing pass — unrecognized gate names (a typo like `'crx'` instead of `'crz'`, or any gate not in `GATE_IDS`) are silently dropped everywhere in the simulator instead of raising, same pattern already documented for the Qiskit interop bridge's unsupported custom gates. Not fixed here — would be a breaking-change decision for `run_circuit`/`run_circuit_jit_beast_mode`/`run_parametric_batch_jit`'s public behavior, tracked separately.

### v8.1.19 — Security fix
- **Fixed (security)**: `QASMParser`'s gate-parameter evaluator (`_eval_param`, used for expressions like `rx(...)`, `p(...)`) called `eval()` with `{'__builtins__': {}}` as its only protection. That blocks direct builtin names (`open`, `len`, `__import__`, ...) but does **not** block attribute/dunder traversal of the live Python object graph (`().__class__.__bases__[0].__subclasses__()...`), which needs no builtin name at all — from there, any class loaded in the process is reachable, including ones whose `__globals__` reference `os`/`subprocess`. Verified directly: a crafted gate-parameter expression, passed through the public `QASMParser.parse()` entry point (the primary entry point of the whole library — used by the dashboard, the Qiskit/PennyLane interop bridge, and any direct usage), executed successfully. Anyone parsing untrusted QASM text was affected, in every previously published version. Fixed by replacing `eval()` with an AST node-type whitelist evaluator (`_eval_ast_node`) — only literals, `+-*/%**` arithmetic, and calls/lookups restricted to the documented math environment (`pi`, `sin`, `sqrt`, ...) are ever evaluated; an `ast.Attribute` node (produced by any `.` in the expression) is never one of the handled cases, so attribute-based escapes are structurally impossible rather than blocklisted. `_resolve_int_expr` (QASM3 `for`-loop bounds) used the same `eval()` pattern but was already protected by a pre-filter regex rejecting any non-arithmetic character — verified safe before this fix — now shares the same AST evaluator for consistency, so no raw `eval()`/`exec()` remains anywhere in the codebase (confirmed via full-repo search). No public API or behavior change for legitimate expressions — every previously-supported parameter syntax (`pi`, `pi/2`, `sqrt(2)`, `cos(0.3)`, etc.) evaluates identically.

**If you parse QASM text from any source you don't fully trust, upgrade immediately.**

### v8.1.18
- **Fixed**: removed a global `warnings.filterwarnings('ignore')` from `registry.py`, run unconditionally on `import dense_evolution`. It silenced every Python warning process-wide for the importing user's whole session — not just this package's, but their own code's and every other library's too. Inherited unchanged from the original Colab notebook (added in v8.0.6, never reconsidered once this became a real pip package). Concretely masked real signal: the JAX float64→float32 truncation `UserWarning`s visible throughout this project's own test output (precision silently lost under `use_float32=True`) would have been invisible to anyone using the package normally.

### v8.1.17
- **Added**: `donate_argnums=(0,)` on `run_circuit_jit_beast_mode`'s statevector buffer — the only one of `_compile_and_run_circuit_jit`'s four call sites where it's safe (`self.sv` is always rebound immediately after, verified across chunked/repeated calls and separate simulator instances). `run_parametric_batch_jit` (its `init_sv` is a `vmap`-broadcast closure shared across the whole batch) and `circuit_to_energy_fn`'s VQE loop (same `stato_zero` reused every epoch) are deliberately left un-donated — donating there would make JAX raise on the second use instead of helping. Verified with a real measurement, not just a claim: RSS growth on a 22-qubit/300-gate circuit drops from +89.4MB to +4.5MB.

### v8.1.16
- **Note**: v8.1.15's published PyPI package does **not** contain the `from_pennylane` Python 3.10 fix described below, despite the changelog entry — the fix landed in the repo before the PyPI upload, but the actual `pip install`-able wheel/sdist for 8.1.15 was built and uploaded from an earlier commit. PyPI doesn't allow re-uploading files under an already-published version, so this release exists specifically to ship that fix as an installable package. If you're on 8.1.15, upgrade to 8.1.16 — don't rely on 8.1.15's changelog matching what you actually have installed.

### v8.1.15
- **Added**: `dense_evolution.autodiff.circuit_to_energy_fn(circuit, n_qubits)` — the real VQE gradient engine (`jax.value_and_grad` through a `jax.lax.scan` circuit template, verified against finite differences to ~1e-11) is now public API, independent of `dashboard_core.py`/Streamlit. Takes a `QASMCircuit` — the same type `from_qiskit`/`from_pennylane` return — so it closes the non-differentiability gap documented in v8.1.14: `circuit_to_energy_fn(from_pennylane(qnode, ...), n_qubits)` now gives a real, non-zero `jax.grad`, verified directly, where `run_pennylane_circuit` alone silently returned `0.0`.
- **Changed**: `dashboard_core.py`'s `_build_vqe_template`/`_vqe_energy_fn` removed — `_run_vqe_telemetry_body` now calls the same public `circuit_to_energy_fn`, one engine instead of two copies of the same math that could silently drift apart. Verified behaviorally identical: all existing dashboard VQE tests pass unchanged, same tolerances.
- **Fixed**: found while testing the newly-public API — calling the engine on a circuit with zero parametric gates crashed on empty-array indexing during JAX tracing. Previously unreachable because `dashboard_core.py` always special-cased `n_params == 0` before calling in; a real gap once this became public API someone could call directly. Fixed with a static (non-traced) branch.
- **Docs**: the README's VQE Engine section had drifted stale, still describing the deleted `risolvi_qasm()` mechanism from before the real-gradient rewrite — corrected, and a new "Differentiable Circuits" section documents `circuit_to_energy_fn` with a verified end-to-end example.
- **Fixed**: `from_pennylane` broke on Python 3.10 — CI caught it (3.10 job red, 3.11/3.12 green). Newer PennyLane releases dropped Python 3.10 support, so pip resolves an older PennyLane (0.42.3) there instead of the version this bridge was built against (0.45.1); `qml.to_openqasm(tape)` behaves incompatibly between the two for a bare tape/QuantumScript input (crashes with `AttributeError: 'QuantumTape' object has no attribute 'func'` on the older one). Verified against both versions directly (installed 0.42.3 in an isolated venv to reproduce). `from_pennylane` now picks whichever serialization path the installed PennyLane version actually supports instead of assuming the newer one unconditionally.

### v8.1.14
- **Added**: interop bridge for Qiskit and PennyLane — `from_qiskit`/`from_pennylane` convert an existing circuit to a `QASMCircuit` by reusing the existing `QASMParser` (via `qiskit.qasm2.dumps` / `qml.to_openqasm`) instead of a bespoke gate-by-gate translator; `run_qiskit_circuit`/`run_pennylane_circuit` execute it directly on `DenseSVSimulator`. Handles the bit-order mismatch explicitly instead of leaving it as a silent trap: Qiskit indexes arrays little-endian (qubit 0 = LSB), Dense-Evolution is MSB-first everywhere else in the codebase, so `run_qiskit_circuit` reorders its output to match Qiskit's own convention (verified against `Statevector(...).probabilities()` on an asymmetric circuit); PennyLane's own wire order already matches Dense-Evolution's natively (verified the same way), so `run_pennylane_circuit` does **not** reorder — kept as two separate code paths on purpose. New optional extras `dense-evolution[qiskit]` / `dense-evolution[pennylane]`.
- **Fixed**: found while building the Qiskit bridge — `qiskit.qasm2.dumps` exports composite gates (e.g. `mcx`) as a `gate NAME params { ... }` definition on a single line, the same brace-delimited block corruption already fixed for QASM3 `for`/`if`/`while`/`def` in v8.1.13, just not covered because `gate` wasn't in that fix's keyword set (verified: before the fix, a 4-qubit circuit using `mcx` silently inflated to `n_qubits=5` with a ghost op). Extended the same brace-matching preprocessor to also strip `gate` definitions cleanly.

### v8.1.13
- **Fixed**: `QASMParser` declared OpenQASM 3.0 support but `for`/`if`/`while`/`def` blocks — brace-delimited, not `;`-terminated — were mishandled by the naive `split(';')` statement splitter: a `for`-loop's body was never extracted, and its closing `}` merged into whatever real statement followed on the same line, corrupting it too (verified: `for int i in [0:2] { h q[i]; } cx q[0],q[1];` produced a single ghost op named `'}'`, with the loop body lost and the real `cx` silently dropped — executed circuit stayed `|000⟩` at 100% probability, no error). Needed for writing VQE ansätze with a loop over qubits instead of one line per qubit. Added `_process_block_constructs`, run before the `;`-split: `for`-loops with resolvable integer bounds (literals, or `int`/`const int` variables declared earlier in the source — QASM3's inclusive-end range semantics) are now genuinely unrolled by substituting the loop variable into the body per iteration; `if`/`while`/`def` blocks and `for`-loops with unresolvable bounds are cleanly stripped instead of corrupting the source that follows them.

### v8.1.12
- **Fixed**: `run_circuit_jit_beast_mode` / `run_parametric_batch_jit` silently dropped `cy`, `cp`, `crz`, `u1`, `p`, `sx` — they weren't in `GATE_IDS`, so `if name not in GATE_IDS: continue` skipped them with no error (verified: `h(0);h(1);crz(0,1,1.2)` produced the exact same output as `h(0);h(1)` alone). `dashboard_core.py` already treats these as first-class gates, so any circuit using them — dashboard-built or hand-written QASM — silently ran the wrong physics through the fast path nearly everything uses. Added the missing `GATE_IDS` entries and the missing kernel implementations for `cy`/`crz`/`sx` in `_apply_gate_fast_step` — `crz` specifically needed its own kernel, not reuse of `cp`'s (CP phases `|11⟩` only; CRZ phases the target conditioned on its own bit, a different gate).
- **Fixed**: `run_circuit_jit_beast_mode` used the raw qubit index as bit position (LSB-first) instead of the documented MSB-first convention (`phys = n_qubits - 1 - qubit`) used by `run_circuit()`/`apply_gate_1q()`/`apply_gate_2q()`/`measure()` elsewhere in the simulator. Pre-existing, not introduced by the fix above — found while verifying it, masked until now because every beast-mode circuit tested to date happened to be symmetric under qubit reversal (Bell states, GHZ states, uniform superpositions), so the wrong labeling never showed up in the probabilities. Verified with `X` on qubit 0 in a 3-qubit register: gave index 1 (LSB) instead of index 4 (MSB, correct). `do_1q`/`do_2q` now compute physical bit positions consistently with the rest of the simulator; `Chunk`'s `num_chunks==1` (via beast mode) and `num_chunks>1` (via `apply_gate_1q`/`apply_gate_2q`) paths are now finally consistent with each other too.
- **Fixed**: the VQE gradient (`run_vqe_telemetry`) was never a real derivative — `grad_vqe_params[i] = 0.5*(energy-target)*sin(theta[i]) + gaussian_noise`, no `jax.grad`, no parameter-shift rule, no backprop on θ anywhere in the codebase (the only real `jax.value_and_grad` usage, in `QMMMForceEngine`, differentiates classical QM/MM forces w.r.t. atomic positions, not circuit parameters). `risolvi_qasm` (the old circuit-building path) converted θ to a Python `float` before use, severing the JAX trace, so backprop couldn't pass through it. Replaced with a real `jax.grad` pipeline reusing `run_parametric_batch_jit`'s own sentinel-injection pattern (θ substituted via `jnp.where` inside a `jax.lax.scan`, never a `float()` call) — verified against a finite-difference gradient (~1.5e-10 agreement) on a real circuit from `QASM_LIBRARY`, and confirmed genuine Adam-optimizer convergence (monotonic energy descent to a minimum) over 40 epochs, unlike the old noisy formula. Public signature and DataFrame columns of `run_vqe_telemetry` unchanged.

### v8.1.11
- **Fixed**: `dash.py` (the original Colab notebook) was declared as an installable module (`py-modules = ["dash"]`) with the *same name* as the real Plotly `dash` package, itself listed as an optional dependency in the very same `pyproject.toml` — a genuine packaging collision, not just a local dev annoyance. It also had unconditional module-level `from google.colab import files` / `import ipywidgets`, so `import dash` crashed immediately outside Colab. Nothing in the maintained codebase (`dashboard_core.py`/`app_dashboard.py`, the real Streamlit port) imports it anymore. Moved to `legacy/dash.py` (reference only, not packaged), removed from `py-modules`. The `dashboard` extra now installs what the real dashboard actually needs (`streamlit`, `pandas`, `seaborn`, `plotly`) instead of the unused `dash` package.
- **Docs**: README's Quick Start (the very first example in the file) passed `circuit.ops` — raw dicts — to `run_circuit_jit_beast_mode`, which expects the tuple format from `circuit.to_tuples()`; crashed with `KeyError: 0`. Fixed, and the "Dashboard" quick-start snippet now points at `streamlit run app_dashboard.py` instead of the retired Colab-only `import dash` pattern.

### v8.1.10
- **Fixed**: `run_circuit_jit_beast_mode` / `run_parametric_batch_jit` — a gate referencing a qubit index out of range silently corrupted the entire statevector to zero instead of raising (verified: `get_probabilities().sum()` went from 1.0 to 0.0, no exception). `apply_gate_1q`/`apply_gate_2q` already validated qubit indices, but these two JIT fast paths build their own compiled ops and never called them. Both now validate before dispatch, matching the existing behavior of the non-JIT path.
- **Fixed**: `Chunk` — for `n_qubits` beyond the RAM-safe budget (`chunk_size_bits`), it silently ran the circuit on a smaller inner simulator (`min(n_qubits, chunk_size_bits)`) instead of genuinely chunking: `num_chunks`/`chunk_dim` were computed but never used to combine multiple pieces. Found testing `Chunk(n_qubits=28)`: `get_probabilities()` returned `2**27` elements, not `2**28`. Now implements real multi-chunk simulation (RAM-only, no disk paging — covers moderate overflow beyond the safe budget, not arbitrarily large qubit counts): `num_chunks` independent chunk-sized simulators held in memory, with gate dispatch across chunk boundaries for all six local/chunk-select combinations. Verified against a plain `DenseSVSimulator` running the identical circuit (exact match, not just "looks right"). A sized RAM check now raises `MemoryPressureError` up front if the chunks wouldn't fit, instead of attempting and OOMing.

### v8.1.9
- **Fixed**: `ia_utils/vector_healing.py` — `enhanced_dense_healing_hybrid` had an unreachable third branch (a dense/blend fallback): the underlying `trigger` signal from `evaluate_phi_trigger` is strictly binary (0.0/1.0), so the branch could never execute. Collapsed to the genuine 2-state logic (pass-through vs. median fallback); runtime output is unchanged since the branch never ran.
- **Fixed**: `dashboard_core.py` — `run_simulation` / `run_vqe_telemetry` mutated the process-wide JAX `jax_enable_x64` flag without ever restoring it, so running one float32 simulation silently downgraded numerical precision for unrelated code later in the same process (e.g. the Vector Healing page, which sets no precision of its own). Both now save/restore the flag around their own execution.
- **Docs**: README's `NoiseModel` example called a nonexistent `.apply()` method with a wrong parameter name (`n_qubits` instead of `n`) — corrected to `apply_to_sv(sv, n=..., ...)`. Documented `QASMCircuit.to_tuples()` and `DenseSVSimulator.run_circuit`, which already existed and work correctly but were never mentioned in the README.

### v8.1.8
- **Fixed**: `parser.py` — controlled two-qubit gates (`cx`/`cy`/`cz`/`cp`/`crz`) parsed from QASM in the dashboard layer had control and target swapped relative to `compiler.py`'s documented `(gate, control, target)` contract, breaking entanglement for circuits run through the dashboard. The core `QASMCircuit.to_tuples()` path was already correct.
- **Fixed**: `parser.py` — range syntax (`q[0:3]`) on single-qubit gates only applied to the first qubit in the range, silently dropping the rest. Now expands into one gate application per qubit, matching the parser's own documented contract.
- **Fixed**: `from dense_evolution import Chunk` raised `ImportError` — `Chunk` is now re-exported from the package root. Added `get_probabilities()`/`get_statevector()` to `Chunk` for parity with `DenseSVSimulator`.
- **Removed**: `dense_evolution/test2.py` and `stress_test.py` — byte-identical, assertion-free debug scripts that shipped inside every install with 0% test coverage. Their one real check (Kraus noise is genuinely stochastic across independent runs) is now a real regression test.

### v8.1.7
- `ia_utils/` — new package: `median_healing`, `enhanced_dense_healing_hybrid` for vector sequence healing (NaN/Inf-safe)
- `jax` import in `ia_utils.vector_healing` made lazy — importable without the `[jax]` extra
- Fixed `reconstruction_error` telemetry returning `NaN` when input contained `NaN`/`Inf`
- Added `scipy` to core dependencies (was used but undeclared)

### v8.1.6
- Modular package structure (`dense_evolution/` directory)
- Split `registry.py`, `gates.py`, `healing.py`, `chunk.py` into dedicated modules

### v8.1.5
- `chunk.py` — `SafeMemoryGuard`: hard block at configurable free-RAM threshold (default 15%), soft warning at 2× threshold, `gc.collect()` before every check
- `chunk.py` — `Chunk` no longer subclasses `DenseSVSimulator`; inner simulator allocated at `safe_qubits` only — eliminates `RESOURCE_EXHAUSTED` on 28q–34q circuits
- `chunk.py` — `CircuitChunker.split_circuit` RAM-checks every gate-slice before dispatch
- `chunk.py` — `MemoryChunker` attributes (`num_chunks`, `chunk_size_bits`, `dtype`) forwarded as `@property` on `Chunk` for benchmark compatibility

### v8.1.0
- `healing.py` — Predictive State Engine: `calculate_phi_ab`, `calculate_vettore_dinamico`, `calculate_delta_preemp`, `evaluate_phi_trigger`, `calculate_jax_reflection` — all `@jax.jit`
- `MemoryReflectionEngine` — event logging + JAX Zero-Drift spectral aggregation

### v8.0.x
- `run_parametric_batch_jit()` — `jax.vmap` over full parameter grids in single XLA call
- `run_circuit_jit_beast_mode()` — static JIT compilation with QuantumTranspiler
- OpenQASM 2.0/3.0 dual-mode parser with paren-depth-aware expression splitting
- `NoiseModel` Kraus channels in `registry.py`

---

## ▍ License

**Business Source License 1.1** — converts automatically to **Apache 2.0** on **1 June 2029**.

- Non-commercial use: unrestricted
- Commercial use: ≤ 24 allocated qubits · ≤ 1,000 circuits/day · ≤ 10,000 shots/circuit
- Attribution required: `© 2026 Salvatore Pennacchio <jtatopenn@libero.it> — Dense Evolution`

Full text: [LICENSE.md](LICENSE.md)

---

<div align="center">
  <sub>© 2026 Salvatore Pennacchio — Dense Evolution</sub>
</div>
