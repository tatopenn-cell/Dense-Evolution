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



\*\*Dense Statevector Quantum Simulator · JAX XLA · NISQ · VQE · QML\*\*



\[!\[CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)

\[!\[PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square\&color=00e5ff)](https://pypi.org/project/dense-evolution/)

\[!\[Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square\&logo=python\&logoColor=white)](https://www.python.org/)

\[!\[License](https://img.shields.io/badge/License-BSL\_1.1-orange?style=flat-square)](LICENSE.md)

\[!\[Build](https://img.shields.io/badge/Build-Passing-00ff9d?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/actions)



\---



\## ▍ What It Is



\*\*Dense Evolution\*\* is a high-performance statevector simulator engineered for deep NISQ circuits, VQE pipelines, and QML workloads. It eliminates Kronecker product overhead entirely via stride-sliced linear kernel fusion compiled through JAX XLA — keeping memory at the theoretical minimum of `2ⁿ × 16 bytes`.



The integrated `dash.py` dashboard provides live ipywidgets telemetry across 6 quantum observables per simulation run, directly inside Google Colab or Jupyter.



\---



\## ▍ Install



```bash

\# core engine

pip install dense-evolution



\# full stack: JAX · GPU · dashboard

pip install dense-evolution\[full]



\# development

git clone https://github.com/tatopenn-cell/Dense-Evolution.git

cd Dense-Evolution \&\& pip install -e .\[full]

```



\*\*Google Colab (3 lines):\*\*

```python

!git clone https://github.com/tatopenn-cell/Dense-Evolution.git

%cd Dense-Evolution

!pip install -e .

```



\---



\## ▍ Quick Start



```python

from dense\_evolution import DenseSVSimulator, QASMParser



\# parse any OpenQASM 2.0 string

qasm = """

OPENQASM 2.0;

include "qelib1.inc";

qreg q\[3];

h q\[0];

cx q\[0], q\[1];

cx q\[1], q\[2];

"""



parser = QASMParser()

circuit = parser.parse(qasm)



sim = DenseSVSimulator(n\_qubits=3)

sim.run\_circuit\_jit\_beast\_mode(circuit.ops)



print(sim.get\_probabilities())   # \[0.5, 0, 0, 0, 0, 0, 0, 0.5]  — GHZ state

print(sim.memory\_mb())           # 0.000128 MB

```



\*\*Dashboard (Colab / Jupyter):\*\*

```python

import dash

from IPython.display import display, clear\_output



clear\_output()

display(dash.dashboard\_unificata)

```



\---



\## ▍ Architecture



```

dense\_evolution/

├── registry.py       hardware detection · JAX / CuPy / NumPy capability flags

├── gates.py          GATES{} · PARAMETRIC\_GATES{} · GATE\_IDS{}

├── noise\_model.py    Kraus channels · stochastic trajectory engine

├── parser.py         QASMParser · QASMCircuit · OpenQASM 2.0 / 3.0

├── compiler.py       \_apply\_gate\_fast\_step (jit) · \_compile\_and\_run\_circuit\_jit

├── simulator.py      DenseSVSimulator · vmap batch VQE · chunked execution

├── chunk.py          SafeMemoryGuard · CircuitChunker · MemoryChunker · Chunk

└── dash.py           ipywidgets dashboard · VQE engine · MD simulation

```



\*\*Data flow per run:\*\*

```

▶ Run

&#x20;└─ core\_calcolo\_quantistico()          parse → JIT execute → apply noise

&#x20;    ├─ ottimizza\_vqe()                 Hellmann-Feynman AD → ADAM → df\_vqe\_telemetry

&#x20;    ├─ run\_md\_simulation\_dummy()       QM/MM dynamics → df\_md\_telemetry + Pearson matrix

&#x20;    └─ build\_panel\_\*(res)              matplotlib figure → display()

```



\---



\## ▍ Core Features



| Feature | Detail |

|---|---|

| \*\*Linear Kernel Fusion\*\* | Stride-sliced tensor ops via JAX XLA — zero Kronecker matrices |

| \*\*Circuit Chunking\*\* | Fixed-size JIT blocks eliminate tracer overhead on 1000+ gate circuits |

| \*\*Anti-OOM SafeMemoryGuard\*\* | Hard block at 15% free RAM — raises `MemoryPressureError` before JAX crashes |

| \*\*Kraus Noise Channels\*\* | `depolarizing` `amplitude\_damping` `phase\_damping` `bitflip` `combined` — stochastic, O(2ⁿ) cost |

| \*\*VQE + ADAM\*\* | Hellmann-Feynman gradient via JAX AD · positional parameter injection into any QASM 2.0 |

| \*\*vmap Batch Sweep\*\* | `run\_parametric\_batch\_jit()` evaluates full parameter grids in one JIT call |

| \*\*Backend Agnostic\*\* | NumPy CPU · JAX XLA CPU/TPU · CuPy CUDA — runtime selection, zero code changes |

| \*\*Live Dashboard\*\* | 8-panel ipywidgets telemetry: probability, VQE energy, entropy, purity, gradient, noise, θ-correction, Pearson heatmap |



\---



\## ▍ Benchmarks



> Measured on Google Colab Free Tier (CPU runtime)



| Metric | Value |

|---|---|

| Numerical drift (80-layer Ansatz, 1360 gates) | `Δ = 1.11 × 10⁻¹⁶` |

| Memory footprint @ 20q | `32 MB` (float64) · `16 MB` (float32) |

| JIT compile overhead (first run) | `< 400 ms` |

| Gate throughput after warm-up | `> 10⁶ gates/s` (CPU) |

| Maximum tested qubits (Colab Free) | `24q` stable · `33q` high-RAM runtime |



\### Anti-OOM Chunk Engine vs PennyLane — Windows CPU (8 GB RAM)



> Dense Evolution maintains constant \~2 GB RAM at any qubit count via dynamic chunking.  

> PennyLane allocates the full statevector — OOM beyond 26q.



| Qubits | Hilbert Space | PennyLane | PennyLane RAM | Dense Evolution | Dense RAM | Chunk Geometry |

|:------:|:-------------:|:---------:|:-------------:|:---------------:|:---------:|:--------------:|

| 24 | 16,777,216 | ✅ SUCCESS | 307 MB | ✅ SUCCESS | 516 MB | 1× (2²⁷) |

| 26 | 67,108,864 | ✅ SUCCESS | 1,074 MB | ✅ SUCCESS | 2,050 MB | 1× (2²⁷) |

| 28 | 268,435,456 | ❌ OOM | — | ✅ SUCCESS | 2,050 MB | 2× (2²⁷) |

| 30 | 1,073,741,824 | ❌ OOM | — | ✅ SUCCESS | 2,048 MB | 8× (2²⁷) |

| 32 | 4,294,967,296 | ❌ OOM | — | ✅ SUCCESS | 2,048 MB | 32× (2²⁷) |



\---



\## ▍ Dashboard Panels



| Panel | Contents |

|---|---|

| \*\*Overview\*\* | R0 header · R1 P(\\|n⟩) histogram + Top-12 states · R2 wavefunction helix 3D + metrics table · R3 noise analysis + shot histogram · R4–R6 VQE telemetry × 6 · R7 Pearson heatmap |

| \*\*Fisica Stato\*\* | Bloch projection · Schmidt rank · coherence vector |

| \*\*Mosaico 1008q\*\* | 2D probability density map up to 1008 qubits |

| \*\*VQE Results\*\* | 6-subplot telemetry: energy convergence, entropy, purity, ‖∇L‖, noise factor, θ-correction |

| \*\*MD Results\*\* | 6-subplot MD telemetry + masked Pearson correlation heatmap |

| \*\*Performance\*\* | Gate throughput · JIT compile time · RAM usage |



\---



\## ▍ Circuit Library (80+ presets)



All circuits are stored as OpenQASM 2.0 strings in `QASM\_LIBRARY`.



\*\*Standard\*\* — Bell Φ⁺, QFT 4q/8q, Toffoli, Adder 2-bit, Deutsch-Jozsa, Bernstein-Vazirani  

\*\*Algorithms\*\* — Grover 3q/4q, Simon 4q, Shor 15, HHL, QAOA Max-Cut 4q, QPE 5q, Quantum Walk, Teleportation, BB84  

\*\*Topological\*\* — Anyonic Braiding 6q, Charge Pump 8q, DiamondPhi 12q, Omega Phase Lock 8q, Arecibo DeepField 16q, ARECIBO v11.3 SINGULARITY  

\*\*Peptide / Biological\*\* — Furin RRAR 8q, Hemoglobin MVLSPADK 8q, Spike 8q/16q, p53 Guardian 24q, WormholeTriplePeptide 24q  

\*\*Stress Tests\*\* — Hardware Stress, Quantum Supremacy, Interference Stress, BGQ 32q, Twin Shield Full Resonance 32q, Nuovo Circuito 33q



\*\*Proprietary phase constants used in topological circuits:\*\*



| Constant | Value (rad) | Physical origin |

|---|:---:|---|

| φ (Golden Ratio) | 1.6180 | Tatopenn φ-resonance |

| sp³ diamond angle | 1.9106 | Carbon tetrahedral bond |

| Topological lock | 3.0718 | Near-π translocation phase |

| Omega / Fe₂S₂ | 6.1574 | Iron-sulfur cluster phase lock |

| BGQ wormhole kick | 0.7000 | BGQ wormhole kickback amplitude |



\---



\## ▍ VQE Engine



\*\*Positional parameter injection\*\* — `QASMParser` tokenizes all literals to `0.0` for JIT speed. VQE recovers parameters by:

1\. counting parametric gates (`rx ry rz p u1 cp crz`) → `n\_params`

2\. initializing `θ ∈ ℝⁿ` uniform in `\[−π, π]`

3\. injecting `θ\[i]` sequentially by gate order in the AST via `risolvi\_qasm()`



Compatible with any custom OpenQASM 2.0 string without pre-labelling.



\*\*Gradient \& update rule:\*\*



$$\\frac{\\partial E}{\\partial \\theta\_i} = \\langle\\psi(\\theta)|\\,\\frac{\\partial H}{\\partial \\theta\_i}\\,|\\psi(\\theta)\\rangle \\qquad \\theta \\leftarrow \\theta - \\frac{\\alpha\\,\\hat{m}\_t}{\\sqrt{\\hat{v}\_t}+\\varepsilon}$$



\*\*Telemetry columns\*\* (→ `df\_vqe\_telemetry`):



| Column | Unit | Description |

|---|---|---|

| `VQE\_Energy` | Ha | ⟨ψ\\|H\\|ψ⟩ |

| `Entropy` | bit | −Tr(ρ log₂ ρ) |

| `Purity` | — | Tr(ρ²) ∈ \[1/d, 1] |

| `Gradient` | — | ‖∇L‖ — barren plateau detection |

| `Noise\_Factor` | — | fidelity-derived noise proxy |

| `Theta\_Correction` | rad | ADAM step norm |



\---



\## ▍ Hamiltonian Library



Auto-filtered by qubit count to prevent shape mismatch.



| Molecule | Qubits | Bond length | E₀ (Ha) |

|---|:---:|:---:|:---:|

| H₂ | 2 | 0.74 Å | −1.13 |

| H₃⁺ | 3 | 0.85 Å | −1.28 |

| LiH | 4 | 1.40 Å | −2.31 |

| H₂O | 5 | 0.96 Å | −4.12 |



Custom: JSON array of diagonal eigenvalues, length `2^n\_qubits`.



\---



\## ▍ Noise Models



All channels applied as post-circuit Kraus operations on the full statevector.



| Model | Kraus operators | Physical process |

|---|---|---|

| `ideal` | I | noiseless |

| `depolarizing` | {√(1−p)I, √(p/3)X,Y,Z} | isotropic Pauli error |

| `amplitude\_damping` | {K₀, K₁} | T₁ energy relaxation |

| `phase\_damping` | {K₀, K₁} | T₂ dephasing |

| `bitflip` | {√(1−p)I, √p·X} | bit flip σₓ |

| `combined` | depolarizing(p/2) ∘ amp\_damp(p/3) | worst-case NISQ |



Fidelity: Bhattacharyya `F = Σᵢ √(pᵢqᵢ)` and TVD `= ½Σᵢ|pᵢ−qᵢ|` computed on every noisy run.



\---



\## ▍ Mitigation \& Predictive Healing Models



Active error tracking and stabilization parameters integrated natively into the simulation runtime.



| Model | Variables / Operators | Physical process |

|---|---|---|

| `dephasing\_tracking` | Δ\_pre\_emp ∘ Σ | predictive deviation vs ideal eigenstate |

| `kappa\_stabilization` | κ-strength routine | proactive statevector profile shielding |

| `richardson\_integration` | {λ₁ = 1.0, λ₂ = 2.0} | dual-point zero-noise trajectory approximation |



Compilation: Full \*\*XLA Kernel Fusion\*\* via `@jax.jit` for mass-parallelized trajectory sweeps (< 1.0s).



\---



\## ▍ Chunk Engines (Anti-OOM)



All operations parcellized dynamically using dual-stage longitudinal and transverse architectural shields.



```python

from dense\_evolution.chunk import Chunk, SafeMemoryGuard



\# Hard block at 15% free RAM — raises MemoryPressureError before JAX crashes

sim = Chunk(n\_qubits=30, memory\_threshold=0.15)

sim.run\_chunk(circuit)



\# Inspect geometry

print(sim.num\_chunks)       # 8

print(sim.chunk\_size\_bits)  # 27

print(sim.memory\_mb())      # \~2048 MB per chunk

```



| Class | Role | Protection |

|---|---|---|

| `SafeMemoryGuard` | RAM monitor | Hard block < 15% free · soft warn < 30% free |

| `CircuitChunker` (`chunk1`) | Circuit regista | RAM check before every gate-slice |

| `MemoryChunker` | Geometry calculator | Reports `num\_chunks`, `chunk\_dim`, `chunk\_size\_bits` |

| `Chunk` (`chunk2`) | Anti-OOM simulator wrapper | Allocates inner sim at `safe\_qubits` only |



\*\*SafeMemoryGuard behaviour:\*\*



| Condition | Action |

|---|---|

| free RAM < `threshold` (default 15%) | `MemoryPressureError` — execution halted cleanly |

| free RAM < `threshold × 2` (default 30%) | Warning printed to stdout |

| Before every gate-slice | `gc.collect()` + RAM check |

| Before inner simulator allocation | Pre-allocation check in `Chunk.\_\_init\_\_` |



Performance: RAM hard-locked at ≤ `chunk\_size\_bits` qubits per allocation block regardless of logical circuit size.



\---



\## ▍ Troubleshooting



| Error | Cause | Fix |

|---|---|---|

| `TypeError: cond branches must have equal output types` | JAX dtype mismatch between 1q/2q branches | `de.patch\_dense\_parametric(de.DenseSVSimulator)` — runs automatically on import |

| `MemoryPressureError` | Free RAM below 15% threshold | Free system RAM · reduce `n\_qubits` · lower `memory\_threshold` |

| `XlaRuntimeError: RESOURCE\_EXHAUSTED` | Using `DenseSVSimulator` directly on > 27q | Use `Chunk(n\_qubits=N)` instead |

| VQE telemetry empty | VQE disabled or no parametric gates | Enable \*\*VQE Settings\*\* checkbox; use circuits with `rx/ry/rz` gates |

| Hamiltonian shape mismatch | JSON array length ≠ 2^n\_qubits | Supply exactly `2^n` values (e.g. 16 for 4q) |

| Barren plateau span not visible | < 3 consecutive epochs with ‖g‖ < 0.01·max‖g‖ | Increase epochs or reduce learning rate |

| Dashboard blank in JupyterLab | Extension missing | `jupyter labextension install @jupyter-widgets/jupyterlab-manager` |



\---



\## ▍ Changelog



\### v8.1.2

\- `chunk.py` — `SafeMemoryGuard` class: hard block at configurable free-RAM threshold (default 15%), soft warning at 2× threshold, `gc.collect()` before every check

\- `chunk.py` — `Chunk` no longer subclasses `DenseSVSimulator`; inner simulator allocated at `safe\_qubits` only — eliminates `RESOURCE\_EXHAUSTED` on 28q–34q circuits

\- `chunk.py` — `CircuitChunker.split\_circuit` RAM-checks every gate-slice before dispatch

\- `chunk.py` — `MemoryChunker` attributes (`num\_chunks`, `chunk\_size\_bits`, `dtype`) forwarded as `@property` on `Chunk` for benchmark compatibility

\- Fixed `globals()\["QuantumTranspiler"]` anti-pattern → direct relative import

\- Fixed `\_\_version\_\_` mismatch between `\_\_init\_\_.py` and PyPI release



\### v8.1.1

\- Modular package structure (`dense\_evolution/` directory)

\- `stress\_test\_quantum.py` deduplicated

\- `test\_library.py` updated with correct imports



\### v8.0.0

\- Initial public release



\---



\## ▍ License



\*\*Business Source License 1.1\*\* — free for research, academic, and non-commercial use.  

Commercial use requires written permission from the author.  

Contact: \[jtatopenn@libero.it](mailto:jtatopenn@libero.it)



\---



<div align="center">

<sub>Dense Evolution · Salvatore Pennacchio · 2026</sub>

</div>

