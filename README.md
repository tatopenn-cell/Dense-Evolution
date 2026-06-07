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
[![PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSL_1.1-orange?style=flat-square)](LICENSE.md)
[![Build](https://img.shields.io/badge/Build-Passing-00ff9d?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/actions)
[![Cross-Validation CI](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/actions/workflows/ci.yml)

---

## ▍ What It Is

**Dense Evolution** is a high-performance statevector simulator engineered for deep NISQ circuits, VQE pipelines, and QML workloads. It eliminates Kronecker product overhead entirely via stride-sliced linear kernel fusion compiled through JAX XLA — keeping memory at the theoretical minimum of `2ⁿ × 16 bytes`.

The integrated `dash.py` dashboard provides live ipywidgets telemetry across 8 panels per simulation run, directly inside Google Colab or Jupyter.

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
sim.run_circuit_jit_beast_mode(circuit.ops)

probs = sim.get_probabilities()
sv    = sim.get_statevector()
```

**Dashboard (Colab / Jupyter):**

```python
import dash
from IPython.display import display, clear_output

clear_output()
display(dash.dashboard_unificata)
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
├── simulator.py    DenseSVSimulator · run_parametric_batch_jit · vmap batch VQE
└── dash.py         ipywidgets dashboard · VQE engine · QM/MM · MD simulation · 3D wavefunction
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
| **Backend Agnostic** | NumPy CPU · JAX XLA CPU/TPU · CuPy CUDA — runtime selection, zero code changes |
| **Live Dashboard** | 8-panel ipywidgets telemetry: probability, VQE energy, entropy, purity, gradient, noise, θ-correction, Pearson heatmap |

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

`QASMCircuit` fields: `n_qubits`, `n_cbits`, `ops` (list of gate tuples).

### `NoiseModel`

```python
noise = NoiseModel()
noise.apply(sv, model='depolarizing', p=0.01, n_qubits=4, rng=rng)
desc  = NoiseModel.kraus_description('amplitude_damping')
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

**Positional parameter injection** — `QASMParser` tokenizes all literals to `0.0` for JIT speed. VQE recovers parameters by:
1. Counting parametric gates (`rx ry rz p u1 cp crz`) → `n_params`
2. Initializing `θ ∈ ℝⁿ` uniform in `[−π, π]`
3. Injecting `θ[i]` sequentially by gate order in the AST via `risolvi_qasm()`

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
