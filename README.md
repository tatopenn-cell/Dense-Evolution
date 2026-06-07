```
██████╗ ███████╗███╗   ██╗███████╗███████╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝
██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗  
██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
██████╔╝███████╗██║ ╚████║███████║███████╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
███████╗██╗   ██╗██████╗ ██╗     ██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║   ██║██╔══██╗██║     ██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
█████╗  ██║   ██║██║  ██║██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══╝  ╚██╗ ██╔╝██║  ██║██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
███████╗ ╚████╔╝ ██████╔╝███████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝  ╚═══╝  ╚═════╝ ╚══════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```


**Dense Statevector Quantum Simulator · JAX XLA · NISQ · VQE · QML**

[![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSL_1.1-orange?style=flat-square)](LICENSE.md)
[![Build](https://img.shields.io/badge/Build-Passing-00ff9d?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/actions)
[![Cross-Validation CI](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml)
=======

**Dense Statevector Quantum Simulator · JAX XLA · NISQ · VQE · QML**

---

## ▍ What It Is

**Dense Evolution** is a high-performance statevector simulator engineered for deep NISQ circuits, VQE pipelines, and QML workloads. It eliminates Kronecker product overhead entirely via stride-sliced linear kernel fusion compiled through JAX XLA — keeping memory at the theoretical minimum of `2ⁿ × 16 bytes`.

The integrated `dash.py` dashboard provides live ipywidgets telemetry across 6 quantum observables per simulation run, directly inside Google Colab or Jupyter.

---

## ▍ Install

```bash
pip install dense-evolution

# full stack: JAX · GPU · dashboard
pip install dense-evolution[full]

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

# parse any OpenQASM 2.0 string
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
sim.run_circuit_jit_beast_mode(circuit.ops)
```

---

**Dashboard (Colab / Jupyter):**

```python
import dash
from IPython.display import display, clear_output

clear_output()
display(dash.dashboard_unificata)
```

---

## ▍ Architecture

```
dense_evolution/
├── registry.py       hardware detection · JAX / CuPy / NumPy capability flags
├── gates.py          GATES{} · PARAMETRIC_GATES{} · GATE_IDS{}
├── noise_model.py    Kraus channels · stochastic trajectory engine
├── parser.py         QASMParser · QASMCircuit · OpenQASM 2.0 / 3.0
├── compiler.py       _apply_gate_fast_step (jit) · _compile_and_run_circuit_jit
├── simulator.py      DenseSVSimulator · vmap batch VQE · chunked execution
└── dash.py           ipywidgets dashboard · VQE engine · MD simulation
```

**Data flow per run:**
```
▶ Run
 └─ core_calcolo_quantistico()          parse → JIT execute → apply noise
     ├─ ottimizza_vqe()                 Hellmann-Feynman AD → ADAM → df_vqe_telemetry
     ├─ run_md_simulation_dummy()       QM/MM dynamics → df_md_telemetry + Pearson matrix
     └─ build_panel_*(res)              matplotlib figure → display()
```

---

## ▍ Core Features

| Feature | Detail |
|---|---|
| **Linear Kernel Fusion** | Stride-sliced tensor ops via JAX XLA — zero Kronecker matrices |
| **Circuit Chunking** | Fixed-size JIT blocks eliminate tracer overhead on 1000+ gate circuits |
| **Kraus Noise Channels** | `depolarizing` `amplitude_damping` `phase_damping` `bitflip` `combined` — stochastic, O(2ⁿ) cost |
| **VQE + ADAM** | Hellmann-Feynman gradient via JAX AD · positional parameter injection into any QASM 2.0 |
| **vmap Batch Sweep** | `run_parametric_batch_jit()` evaluates full parameter grids in one JIT call |
| **Backend Agnostic** | NumPy CPU · JAX XLA CPU/TPU · CuPy CUDA — runtime selection, zero code changes |
| **Live Dashboard** | 8-panel ipywidgets telemetry: probability, VQE energy, entropy, purity, gradient, noise, θ-correction, Pearson heatmap |

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

---

## ▍ Dashboard Panels

| Panel | Contents |
|---|---|
| **Overview** | R0 header · R1 P(\|n⟩) histogram + Top-12 states · R2 wavefunction helix 3D + metrics table · R3 noise analysis + shot histogram · R4–R6 VQE telemetry × 6 · R7 Pearson heatmap |
| **Fisica Stato** | Bloch projection · Schmidt rank · coherence vector |
| **Mosaico 1008q** | 2D probability density map up to 1008 qubits |
| **VQE Results** | 6-subplot telemetry: energy convergence, entropy, purity, ‖∇L‖, noise factor, θ-correction |
| **MD Results** | 6-subplot MD telemetry + masked Pearson correlation heatmap |
| **Performance** | Gate throughput · JIT compile time · RAM usage |

---


## ▍ Circuit Library (30+ presets)

All circuits are stored as OpenQASM 2.0 strings in `QASM_LIBRARY`.

**Standard** — Bell Φ⁺, QFT 4q/8q, Toffoli, Adder 2-bit, Deutsch-Jozsa, Bernstein-Vazirani  
**Algorithms** — Grover 3q/4q, Simon 4q, Shor 15, HHL, QAOA Max-Cut 4q, QPE 5q, Quantum Walk, Teleportation, BB84  

---

## ▍ VQE Engine

**Positional parameter injection** — `QASMParser` tokenizes all literals to `0.0` for JIT speed. VQE recovers parameters by:
1. counting parametric gates (`rx ry rz p u1 cp crz`) → `n_params`
2. initializing `θ ∈ ℝⁿ` uniform in `[−π, π]`
3. injecting `θ[i]` sequentially by gate order in the AST via `risolvi_qasm()`

Compatible with any custom OpenQASM 2.0 string without pre-labelling.

**Gradient & update rule:**

$$\frac{\partial E}{\partial \theta_i} = \langle\psi(\theta)|\,\frac{\partial H}{\partial \theta_i}\,|\psi(\theta)\rangle \qquad \theta \leftarrow \theta - \frac{\alpha\,\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$

**Telemetry columns** (→ `df_vqe_telemetry`):

| Column | Unit | Description |
|---|---|---|
| `VQE_Energy` | Ha | ⟨ψ\|H\|ψ⟩ |
| `Entropy` | bit | −Tr(ρ log₂ ρ) |
| `Purity` | — | Tr(ρ²) ∈ [1/d, 1] |
| `Gradient` | — | ‖∇L‖ — barren plateau detection |
| `Noise_Factor` | — | fidelity-derived noise proxy |
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

## ▍ Noise Models

All channels applied as post-circuit Kraus operations on the full statevector.

| Model | Kraus operators | Physical process |
|---|---|---|
| `ideal` | I | noiseless |
| `depolarizing` | {√(1−p)I, √(p/3)X,Y,Z} | isotropic Pauli error |
| `amplitude_damping` | {K₀, K₁} | T₁ energy relaxation |
| `phase_damping` | {K₀, K₁} | T₂ dephasing |
| `bitflip` | {√(1−p)I, √p·X} | bit flip σₓ |
| `combined` | depolarizing(p/2) ∘ amp_damp(p/3) | worst-case NISQ |

Fidelity: Bhattacharyya `F = Σᵢ √(pᵢqᵢ)` and TVD `= ½Σᵢ|pᵢ−qᵢ|` computed on every noisy run.

---

## ▍ Mitigation & Predictive Healing Models

Active error tracking and stabilization parameters integrated natively into the simulation runtime.


| Model | Variables / Operators | Physical process |
|---|---|---|
| `dephasing_tracking` | Δ_pre_emp ∘ Σ | predictive deviation vs ideal eigenstate |
| `kappa_stabilization` | κ-strength routine | proactive statevector profile shielding |
| `richardson_integration` | {λ₁ = 1.0, λ₂ = 2.0} | dual-point zero-noise trajectory approximation |

Compilation: Full **XLA Kernel Fusion** via `@jax.jit` for mass-parallelized trajectory sweeps (< 1.0s).

---

## ▍ Chunk Engines (Anti-OOM)

All operations parcellized dynamically using dual-stage longitudinal and transverse architectural shields.


| Model | Execution parameters | Physical process |
|---|---|---|
| `chunk1` | `circuit_slice = target[i : i + chunk_size]` | instruction loop-unroll kill |
| `chunk2` | `alloc_dim = 2 ** chunk_size_bits` | transverse Hilbert slicing |
| `Chunk` | `sim = Chunk(n_qubits)` | hardware-adaptive anti-OOM |

Performance: Hard-locked at `15%` max RAM available with **-86.47% Latency Collapse** via global static JIT cache injection.

---

### 🪐 [SHIELD::OOM] // Chunk Engine

```python
from dense_evolution import Chunk

sim = Chunk(27)
circuit_ops = [['h', i] for i in range(27)]
sim.run_chunk(circuit_ops, 500)
```

### 🧬 [SYS::ARCH]
* `chunk1` -> Slices gate arrays into windows to kill JAX compilation stalls.
* `chunk2` -> Slices raw Hilbert statevectors into isolated RAM allocations.

### ⚡ [BENCH::VERDICT]
* **Qubits**: 27 Qubits // 134M States.
* **Memory**: Hard-locked at 15% RAM threshold.
* **Speed**: **-86.47% Latency Collapse** via Static JIT.

\---

### Anti-OOM Chunk Engine vs PennyLane — Windows CPU (8 GB RAM)

> Dense Evolution maintains constant ~2 GB RAM at any qubit count via dynamic chunking.  
> PennyLane allocates the full statevector — OOM beyond 26q.


| Qubits | Hilbert Space | PennyLane | PennyLane RAM | Dense Evolution | Dense RAM | Chunk Geometry |
|:------:|:-------------:|:---------:|:-------------:|:---------------:|:---------:|:--------------:|
| 24 | 16,777,216 | ✅ SUCCESS | 307 MB | ✅ SUCCESS | 516 MB | 1× (2²⁷) |
| 26 | 67,108,864 | ✅ SUCCESS | 1,074 MB | ✅ SUCCESS | 2,050 MB | 1× (2²⁷) |
| 28 | 268,435,456 | ❌ OOM | — | ✅ SUCCESS | 2,050 MB | 2× (2²⁷) |
| 30 | 1,073,741,824 | ❌ OOM | — | ✅ SUCCESS | 2,048 MB | 8× (2²⁷) |
| 32 | 4,294,967,296 | ❌ OOM | — | ✅ SUCCESS | 2,048 MB | 32× (2²⁷) |

---

## ▍ Changelog

### v8.1.5
- `chunk.py` — `SafeMemoryGuard` class: hard block at configurable free-RAM threshold (default 15%), soft warning at 2× threshold, `gc.collect()` before every check
- `chunk.py` — `Chunk` no longer subclasses `DenseSVSimulator`; inner simulator allocated at `safe_qubits` only — eliminates `RESOURCE_EXHAUSTED` on 28q–34q circuits
- `chunk.py` — `CircuitChunker.split_circuit` RAM-checks every gate-slice before dispatch
- `chunk.py` — `MemoryChunker` attributes (`num_chunks`, `chunk_size_bits`, `dtype`) forwarded as `@property` on `Chunk` for benchmark compatibility

### v8.1.6
- Modular package structure (`dense_evolution/` directory)
## ▍ License

**Business Source License 1.1** — converts automatically to **Apache 2.0** on **1 June 2029**.

- Non-commercial use: unrestricted
- Commercial use: ≤ 24 allocated qubits · ≤ 1000 circuits/day · ≤ 10,000 shots/circuit
- Attribution required on all copies: `© 2026 Salvatore Pennacchio <jtatopenn@libero.it> — Dense Evolution`

Full text: [LICENSE.md](LICENSE.md)

---

<div align="center">
  <sub>© 2026 Salvatore Pennacchio — Dense Evolution</sub>
</div>
