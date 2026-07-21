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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatopenn-cell/Dense-Evolution/blob/main/notebooks/dashboard_colab.ipynb)
[![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
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

A Streamlit dashboard (`app_dashboard.py`) provides live telemetry across 8 panels per simulation run — Quantum Simulator and Vector Healing tabs, run locally with `streamlit run app_dashboard.py`. `legacy/dash.py` is the original Google Colab notebook this was ported from, kept for reference only (not installable — see the file header).

---

## ▍ Install

```bash
pip install dense-evolution

# full stack: JAX · GPU · dashboard · Qiskit/PennyLane interop
pip install dense-evolution[full]

# just the interop bridge
pip install dense-evolution[qiskit]
pip install dense-evolution[pennylane]

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
sim.run_circuit_jit_beast_mode(circuit.to_tuples())

probs = sim.get_probabilities()
sv    = sim.get_statevector()
```

**Dashboard (local, Streamlit):**

```bash
pip install "dense-evolution[jax,dashboard]"
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
├── registry.py     hardware detection · JAX/CuPy/NumPy flags · NoiseModel (Kraus channels)
├── gates.py        GATES{} · PARAMETRIC_GATES{} · GATE_IDS{}
├── healing.py      predictive state engine · Phi_AB · vettore dinamico · MemoryReflectionEngine
├── parser.py       QASMParser · QASMCircuit · OpenQASM 2.0 / 3.0
├── compiler.py     QuantumTranspiler · _apply_gate_fast_step (jit) · gate decomposition
├── chunk.py        SafeMemoryGuard · MemoryChunker · CircuitChunker · Chunk (Anti-OOM)
└── simulator.py    DenseSVSimulator · run_parametric_batch_jit · vmap batch VQE

ia_utils/
└── vector_healing.py   median_healing · enhanced_dense_healing_hybrid (NaN/Inf-safe, lazy JAX import)

app_dashboard.py + dashboard_core.py + ui_pages/   Streamlit dashboard — VQE engine · QM/MM · MD simulation · 3D wavefunction
legacy/dash.py                                     original Colab notebook, reference only (not installed as a module)
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
| **Parametric Batch JIT** | `run_parametric_batch_jit()` evaluates full parameter grids in one `jax.vmap` + `jax.jit` call |
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
| `run_circuit_jit_beast_mode(circuit)` | JIT-compiled gate execution — primary execution path |
| `run_circuit_with_chunking(circuit, chunk_size=500)` | Chunked execution for long circuits |
| `run_parametric_batch_jit(base_circuit, parameter_batch)` | `vmap` over parameter grid — returns full batch of statevectors |
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
format that `run_circuit` / `run_circuit_jit_beast_mode` expect — don't
build that format by hand.

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
healed, metadata = enhanced_dense_healing_hybrid(vettori, radius_baseline=None, median_fallback_threshold=0.1)
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

The real VQE gradient engine (`jax.value_and_grad` through a `jax.lax.scan`-based circuit template, verified against finite differences to ~1e-11) used to live only inside `dashboard_core.py`, unreachable from outside the Streamlit app. It's now public, dependency-light (needs only `dense-evolution[jax]`, no dashboard/pandas/streamlit), and composes directly with the interop bridge above:

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

---

## ▍ Mitigation & Predictive Healing

Active error tracking and stabilization integrated natively into the simulation runtime via `healing.py`.

| Model | Operators | Description |
|---|---|---|
| `dephasing_tracking` | `Δ_pre_emp ∘ Σ` | Predictive deviation vs ideal eigenstate |
| `phi_ab_alignment` | `Φ_AB(state_A, state_B, ipg)` | Semantic + coherence alignment between two quantum states |
| `vettore_dinamico` | `V_din = K · log(E_B/E_A) · Φ_AB` | Log-differential energetic evolution vector |
| `kappa_stabilization` | `κ-strength routine` | Proactive statevector profile shielding |
| `richardson_integration` | `{λ₁=1.0, λ₂=2.0}` | Dual-point zero-noise trajectory approximation |

All core functions compiled via `@jax.jit`. Event history managed by `MemoryReflectionEngine` with JAX Zero-Drift spectral aggregation.

---

## ▍ IA Utils — Vector Sequence Healing

`ia_utils/vector_healing.py` — standalone module for cleaning sequences of vectors (e.g. hidden states / embeddings) that may contain `NaN` or `Inf` entries. Both functions preprocess the input (Inf → NaN → column-mean imputation) before healing, so corrupted values never propagate into the output.

| Function | Approach | Returns |
|---|---|---|
| `median_healing(vettori, radius_baseline=None)` | `scipy.ndimage.median_filter`, dynamic radius `min(20, max(3, n // 3))` | `(healed: np.ndarray, radius: int)` |
| `enhanced_dense_healing_hybrid(vettori, radius_baseline=None, median_fallback_threshold=0.1)` | Blends the `dense_evolution.healing` Φ-trigger logic with a median fallback, decided per-step | `(healed: np.ndarray, metadata: dict)` |

`enhanced_dense_healing_hybrid` metadata:

| Key | Type | Description |
|---|---|---|
| `fallback_triggered` | `bool` | `True` if the median fallback or dense blending fired at least once |
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

`jax` is imported lazily inside `enhanced_dense_healing_hybrid` — `median_healing` and the module import itself work without the `[jax]` extra installed; only calling `enhanced_dense_healing_hybrid` requires it.

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

Auto-filtered by qubit count to prevent shape mismatch.

| Molecule | Qubits | Bond length | E₀ (Ha) |
|---|:---:|:---:|:---:|
| H₂ | 2 | 0.74 Å | −1.13 |
| H₃⁺ | 3 | 0.85 Å | −1.28 |
| LiH | 4 | 1.40 Å | −2.31 |
| H₂O | 5 | 0.96 Å | −4.12 |

Custom: JSON array of diagonal eigenvalues, length `2^n_qubits`.

---

## ▍ Circuit Library (30+ presets)

All circuits stored as OpenQASM 2.0 strings in `QASM_LIBRARY`.

**Standard** — Bell Φ⁺, QFT 4q/8q, Toffoli, Adder 2-bit, Deutsch-Jozsa, Bernstein-Vazirani

**Algorithms** — Grover 3q/4q, Simon 4q, Shor 15, HHL, QAOA Max-Cut 4q, QPE 5q, Quantum Walk, Teleportation, BB84

---

## ▍ Changelog

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
