```
██████╗ ███████╗███╗   ██╗███████╗███████╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝
██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗  
██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
██████╔╝███████╗██║ ╚████║███████║███████╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
███████╗██╗   ██╗ ██████╗ ██╗     
██╔════╝██║   ██║██╔═══██╗██║     
█████╗  ██║   ██║██║   ██║██║     
██╔══╝  ╚██╗ ██╔╝██║   ██║██║     
███████╗ ╚████╔╝ ╚██████╔╝███████╗
╚══════╝  ╚═══╝   ╚═════╝ ╚══════╝
```

**Dense Statevector Quantum Simulator · JAX XLA · NISQ · VQE · QML**

[![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSL_1.1-orange?style=flat-square)](LICENSE.md)
[![Build](https://img.shields.io/badge/Build-Passing-00ff9d?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/actions)

</div>

---

## ▍ What It Is

**Dense Evolution** is a high-performance statevector simulator engineered for deep NISQ circuits, VQE pipelines, and QML workloads. It eliminates Kronecker product overhead entirely via stride-sliced linear kernel fusion compiled through JAX XLA — keeping memory at the theoretical minimum of `2ⁿ × 16 bytes`.

The integrated `dash.py` dashboard provides live ipywidgets telemetry across 6 quantum observables per simulation run, directly inside Google Colab or Jupyter.

---

## ▍ Install

```bash
# core engine
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

print(sim.get_probabilities())   # [0.5, 0, 0, 0, 0, 0, 0, 0.5]  — GHZ state
print(sim.memory_mb())           # 0.000128 MB
```

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
| Numerical drift (80-layer Ansatz, 1360 gates) | `Δ = 1.11 × 10⁻¹⁶` |
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
**Topological** — Anyonic Braiding 6q, Charge Pump 8q   
**Stress Tests** — Hardware Stress, Quantum Supremacy, Interference Stress


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

## ▍ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `TypeError: cond branches must have equal output types` | JAX dtype mismatch between 1q/2q branches | `de.patch_dense_parametric(de.DenseSVSimulator)` — runs automatically on import |
| VQE telemetry empty | VQE disabled or no parametric gates | Enable **VQE Settings** checkbox; use circuits with `rx/ry/rz` gates |
| Hamiltonian shape mismatch | JSON array length ≠ 2^n_qubits | Supply exactly `2^n` values (e.g. 16 for 4q) |
| Barren plateau span not visible | < 3 consecutive epochs with ‖g‖ < 0.01·max‖g‖ | Increase epochs or reduce learning rate |
| Dashboard blank in JupyterLab | Extension missing | `jupyter labextension install @jupyter-widgets/jupyterlab-manager` |
| Memory error on high-qubit circuits | 2ⁿ × 16 bytes: 24q = 268 MB, 30q = 16 GB | Use `use_float32=True` to halve; cap at 24q on standard runtimes |

---

## ▍ License

**Business Source License 1.1** — converts automatically to **Apache 2.0** on **1 June 2029**.

- Non-commercial use: unrestricted
- Commercial use: ≤ 24 allocated qubits · ≤ 1000 circuits/day · ≤ 10,000 shots/circuit
- Attribution required on all copies: `© 2026 Salvatore Pennacchio <jtatopenn@libero.it> — Dense Evolution`

Full text: [LICENSE.md](LICENSE.md)

---

<div align="center">
<sub>© 2026 Salvatore Pennacchio — Dense Evolution v8.0.7</sub>

