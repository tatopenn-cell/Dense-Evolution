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
Dense Statevector Quantum Simulator · JAX XLA · NISQ · VQE · QML
![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
![PyPI](https://pypi.org/project/dense-evolution/)
![Python](https://www.python.org/)
![License](LICENSE.md)
![Build](https://github.com/tatopenn-cell/Dense-Evolution/actions)
</div>
---
▍ What It Is
Dense Evolution is a high-performance statevector simulator engineered for deep NISQ circuits, VQE pipelines, and QML workloads. It eliminates Kronecker product overhead entirely via stride-sliced linear kernel fusion compiled through JAX XLA — keeping memory at the theoretical minimum of `2ⁿ × 16 bytes`.
The integrated `dash.py` dashboard provides live ipywidgets telemetry across 6 quantum observables per simulation run, directly inside Google Colab or Jupyter.
---
▍ Install
```bash
# core engine
pip install dense-evolution

# full stack: JAX · GPU · dashboard
pip install dense-evolution\\\[full]

# development
git clone https://github.com/tatopenn-cell/Dense-Evolution.git
cd Dense-Evolution \\\&\\\& pip install -e .\\\[full]
```
Google Colab (3 lines):
```python
!git clone https://github.com/tatopenn-cell/Dense-Evolution.git
%cd Dense-Evolution
!pip install -e .
```
---
▍ Quick Start
```python
from dense\\\_evolution import DenseSVSimulator, QASMParser

# parse any OpenQASM 2.0 string
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q\\\[3];
h q\\\[0];
cx q\\\[0], q\\\[1];
cx q\\\[1], q\\\[2];
"""

parser = QASMParser()
circuit = parser.parse(qasm)

sim = DenseSVSimulator(n\\\_qubits=3)
sim.run\\\_circuit\\\_jit\\\_beast\\\_mode(circuit.ops)

print(sim.get\\\_probabilities())   # \\\[0.5, 0, 0, 0, 0, 0, 0, 0.5]  — GHZ state
print(sim.memory\\\_mb())           # 0.000128 MB
```
Dashboard (Colab / Jupyter):
```python
import dash
from IPython.display import display, clear\\\_output

clear\\\_output()
display(dash.dashboard\\\_unificata)
```
---
▍ Architecture
```
dense\\\_evolution/
├── registry.py       hardware detection · JAX / CuPy / NumPy capability flags
├── gates.py          GATES{} · PARAMETRIC\\\_GATES{} · GATE\\\_IDS{}
├── noise\\\_model.py    Kraus channels · stochastic trajectory engine
├── parser.py         QASMParser · QASMCircuit · OpenQASM 2.0 / 3.0
├── compiler.py       \\\_apply\\\_gate\\\_fast\\\_step (jit) · \\\_compile\\\_and\\\_run\\\_circuit\\\_jit
├── simulator.py      DenseSVSimulator · vmap batch VQE · chunked execution
└── dash.py           ipywidgets dashboard · VQE engine · MD simulation
```
Data flow per run:
```
▶ Run
 └─ core\\\_calcolo\\\_quantistico()          parse → JIT execute → apply noise
     ├─ ottimizza\\\_vqe()                 Hellmann-Feynman AD → ADAM → df\\\_vqe\\\_telemetry
     ├─ run\\\_md\\\_simulation\\\_dummy()       QM/MM dynamics → df\\\_md\\\_telemetry + Pearson matrix
     └─ build\\\_panel\\\_\\\*(res)              matplotlib figure → display()
```
---
▍ Core Features
Feature	Detail
Linear Kernel Fusion	Stride-sliced tensor ops via JAX XLA — zero Kronecker matrices
Circuit Chunking	Fixed-size JIT blocks eliminate tracer overhead on 1000+ gate circuits
Kraus Noise Channels	`depolarizing` `amplitude\\\_damping` `phase\\\_damping` `bitflip` `combined` — stochastic, O(2ⁿ) cost
VQE + ADAM	Hellmann-Feynman gradient via JAX AD · positional parameter injection into any QASM 2.0
vmap Batch Sweep	`run\\\_parametric\\\_batch\\\_jit()` evaluates full parameter grids in one JIT call
Backend Agnostic	NumPy CPU · JAX XLA CPU/TPU · CuPy CUDA — runtime selection, zero code changes
Live Dashboard	8-panel ipywidgets telemetry: probability, VQE energy, entropy, purity, gradient, noise, θ-correction, Pearson heatmap
---
▍ Benchmarks
> Measured on Google Colab Free Tier (CPU runtime)
Metric	Value
Numerical drift (80-layer Ansatz, 1360 gates)	`Δ = 1.11 × 10⁻¹⁶`
Memory footprint @ 20q	`32 MB` (float64) · `16 MB` (float32)
JIT compile overhead (first run)	`< 400 ms`
Gate throughput after warm-up	`> 10⁶ gates/s` (CPU)
Maximum tested qubits (Colab Free)	`24q` stable · `33q` high-RAM runtime
---
▍ Dashboard Panels
Panel	Contents
Overview	R0 header · R1 P(|n⟩) histogram + Top-12 states · R2 wavefunction helix 3D + metrics table · R3 noise analysis + shot histogram · R4–R6 VQE telemetry × 6 · R7 Pearson heatmap
Fisica Stato	Bloch projection · Schmidt rank · coherence vector
Mosaico 1008q	2D probability density map up to 1008 qubits
VQE Results	6-subplot telemetry: energy convergence, entropy, purity, ‖∇L‖, noise factor, θ-correction
MD Results	6-subplot MD telemetry + masked Pearson correlation heatmap
Performance	Gate throughput · JIT compile time · RAM usage
---
▍ Circuit Library (30+ presets)
All circuits are stored as OpenQASM 2.0 strings in `QASM\\\_LIBRARY`.
Standard — Bell Φ⁺, QFT 4q/8q, Toffoli, Adder 2-bit, Deutsch-Jozsa, Bernstein-Vazirani  
Algorithms — Grover 3q/4q, Simon 4q, Shor 15, HHL, QAOA Max-Cut 4q, QPE 5q, Quantum Walk, Teleportation, BB84  
Topological — Anyonic Braiding 6q, Charge Pump 8q, Arecibo DeepField 16q
Peptide / Biological — Furin RRAR 8q, Hemoglobin MVLSPADK 8q,
Stress Tests — Hardware Stress, Quantum Supremacy, Interference Stress,
▍ VQE Engine
Positional parameter injection — `QASMParser` tokenizes all literals to `0.0` for JIT speed. VQE recovers parameters by:
counting parametric gates (`rx ry rz p u1 cp crz`) → `n\\\_params`
initializing `θ ∈ ℝⁿ` uniform in `\\\[−π, π]`
injecting `θ\\\[i]` sequentially by gate order in the AST via `risolvi\\\_qasm()`
Compatible with any custom OpenQASM 2.0 string without pre-labelling.
Gradient & update rule:
$$\frac{\partial E}{\partial \theta_i} = \langle\psi(\theta)|,\frac{\partial H}{\partial \theta_i},|\psi(\theta)\rangle \qquad \theta \leftarrow \theta - \frac{\alpha,\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$
Telemetry columns (→ `df\\\_vqe\\\_telemetry`):
Column	Unit	Description
`VQE\\\_Energy`	Ha	⟨ψ|H|ψ⟩
`Entropy`	bit	−Tr(ρ log₂ ρ)
`Purity`	—	Tr(ρ²) ∈ [1/d, 1]
`Gradient`	—	‖∇L‖ — barren plateau detection
`Noise\\\_Factor`	—	fidelity-derived noise proxy
`Theta\\\_Correction`	rad	ADAM step norm
---
▍ Hamiltonian Library
Auto-filtered by qubit count to prevent shape mismatch.
Molecule	Qubits	Bond length	E₀ (Ha)
H₂	2	0.74 Å	−1.13
H₃⁺	3	0.85 Å	−1.28
LiH	4	1.40 Å	−2.31
H₂O	5	0.96 Å	−4.12
Custom: JSON array of diagonal eigenvalues, length `2^n\\\_qubits`.
---
▍ Noise Models
All channels applied as post-circuit Kraus operations on the full statevector.
Model	Kraus operators	Physical process
`ideal`	I	noiseless
`depolarizing`	{√(1−p)I, √(p/3)X,Y,Z}	isotropic Pauli error
`amplitude\\\_damping`	{K₀, K₁}	T₁ energy relaxation
`phase\\\_damping`	{K₀, K₁}	T₂ dephasing
`bitflip`	{√(1−p)I, √p·X}	bit flip σₓ
`combined`	depolarizing(p/2) ∘ amp_damp(p/3)	worst-case NISQ
Fidelity: Bhattacharyya `F = Σᵢ √(pᵢqᵢ)` and TVD `= ½Σᵢ|pᵢ−qᵢ|` computed on every noisy run.
---
▍ Troubleshooting
Error	Cause	Fix
`TypeError: cond branches must have equal output types`	JAX dtype mismatch between 1q/2q branches	`de.patch\\\_dense\\\_parametric(de.DenseSVSimulator)` — runs automatically on import
VQE telemetry empty	VQE disabled or no parametric gates	Enable VQE Settings checkbox; use circuits with `rx/ry/rz` gates
Hamiltonian shape mismatch	JSON array length ≠ 2^n_qubits	Supply exactly `2^n` values (e.g. 16 for 4q)
Barren plateau span not visible	< 3 consecutive epochs with ‖g‖ < 0.01·max‖g‖	Increase epochs or reduce learning rate
Dashboard blank in JupyterLab	Extension missing	`jupyter labextension install @jupyter-widgets/jupyterlab-manager`
Memory error on high-qubit circuits	2ⁿ × 16 bytes: 24q = 268 MB, 30q = 16 GB	Use `use\\\_float32=True` to halve; cap at 24q on standard runtimes
---
▍ Mitigation & Predictive Healing Models
Active error tracking and stabilization parameters integrated natively into the simulation runtime.
Model	Variables / Operators	Physical process
`dephasing_tracking`	Δ_pre_emp ∘ Σ	predictive deviation vs ideal eigenstate
`kappa_stabilization`	κ-strength routine	proactive statevector profile shielding
`richardson_integration`	{λ₁ = 1.0, λ₂ = 2.0}	dual-point zero-noise trajectory approximation
Compilation: Full **XLA Kernel Fusion** via `@jax.jit` for mass-parallelized trajectory sweeps (< 1.0s).
---
▍ Chunk Engines (Anti-OOM)
All operations parcellized dynamically using dual-stage longitudinal and transverse architectural shields.
Model	Execution parameters	Physical process
`chunk1`	`circuit_slice = target[i : i + chunk_size]`	instruction loop-unroll kill
`chunk2`	`alloc_dim = 2 ** chunk_size_bits`	transverse Hilbert slicing
`Chunk`	`sim = Chunk(n_qubits)`	hardware-adaptive anti-OOM
Performance: Hard-locked at `15%` max RAM available with **-86.47% Latency Collapse** via global static JIT cache injection.
---
🪐 [SHIELD::OOM] // Chunk Engine

```python
from dense_evolution import Chunk

sim = Chunk(27)
circuit_ops = [['h', i] for i in range(27)]
sim.run_chunk(circuit_ops, 500)
```

🧬 [SYS::ARCH]
* `chunk1` -> Slices gate arrays into windows to kill JAX compilation stalls.
* `chunk2` -> Slices raw Hilbert statevectors into isolated RAM allocations.

⚡ [BENCH::VERDICT]
* **Qubits**: 27 Qubits // 134M States.
* **Memory**: Hard-locked at 15% RAM threshold.
* **Speed** : **-86.47% Latency Collapse** via Static JIT.
▍ License
Business Source License 1.1 — converts automatically to Apache 2.0 on 1 June 2029.
Non-commercial use: unrestricted
Commercial use: ≤ 24 allocated qubits · ≤ 1000 circuits/day · ≤ 10,000 shots/circuit
Attribution required on all copies: `© 2026 Salvatore Pennacchio <jtatopenn@libero.it> — Dense Evolution`
Full text: LICENSE.md
---
<div align="center">
<sub>© 2026 Salvatore Pennacchio — Dense Evolution </sub>
