<p align="center">
  <img src="docs/assets/banner.svg" alt="Dense Evolution — NISQ quantum simulation toolkit, JAX-native" width="900">
</p>

<!-- mcp-name: io.github.tatopenn-cell/dense-evolution -->
**A high-performance quantum simulation toolkit
Statevector/MPS engines with compilation, noise, VQE, QEC, chemistry, and agent-native tooling.**

[![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square)](https://tatopenn-cell.github.io/Dense-Evolution/)
[![codecov](https://codecov.io/gh/tatopenn-cell/Dense-Evolution/branch/main/graph/badge.svg)](https://codecov.io/gh/tatopenn-cell/Dense-Evolution)
[![PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSL_1.1-orange?style=flat-square)](LICENSE.md)
[![Build](https://img.shields.io/badge/Build-Passing-00ff9d?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/actions)
[![Cross-Validation CI](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/tatopenn-cell/Dense-Evolution?style=flat-square&color=blueviolet)](https://github.com/tatopenn-cell/Dense-Evolution/releases)
[![Last Commit](https://img.shields.io/github/last-commit/tatopenn-cell/Dense-Evolution?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/commits/main)
[![Issues](https://img.shields.io/github/issues/tatopenn-cell/Dense-Evolution?style=flat-square)](https://github.com/tatopenn-cell/Dense-Evolution/issues)
[![Stars](https://img.shields.io/github/stars/tatopenn-cell/Dense-Evolution?style=flat-square&color=yellow)](https://github.com/tatopenn-cell/Dense-Evolution/stargazers)
[![JAX](https://img.shields.io/badge/Backend-JAX_XLA-f9ab00?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![DOI](https://zenodo.org/badge/1247011090.svg)](https://doi.org/10.5281/zenodo.21855643)
[![Featured in Awesome Quantum Software](https://img.shields.io/badge/Featured%20in-Awesome%20Quantum%20Software-blueviolet?style=flat-square)](https://github.com/qosf/awesome-quantum-software)

---

## Table of Contents
- [What It Is](#-what-it-is)
- [Install](#-install)
- [Quick Start](#-quick-start)
- [Key Features](#-key-features)
- [Benchmarks](#-benchmarks)
- [Composer & MCP Server](#-composer--mcp-server)
- [Key Resources](#-key-resources)

## ▍ What It Is

Run up to 28 qubits in about 3 seconds, without crashing. **Dense Evolution** JIT-compiles statevector circuits through JAX XLA, automatically chunks and — past even that RAM ceiling — spills to disk when memory fills up, so a real simulation stays alive instead of OOM-ing.

📖 **[Full documentation, API reference, and worked examples →](https://tatopenn-cell.github.io/Dense-Evolution/)**

A local Streamlit dashboard and web-based Composer editor are also included — see [Composer & MCP Server](#-composer--mcp-server) below.

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

<details>
<summary>⚠️ macOS + <code>dense-evolution[qiskit]</code> users</summary>

Qiskit's own `QuantumCircuit.__init__` is known to segfault the whole process on macOS/arm64 (an upstream Qiskit bug, not something Dense-Evolution can fix from its side — see [release v8.1.43](https://github.com/tatopenn-cell/Dense-Evolution/releases/tag/v8.1.43) for the full reproduction). `dense_evolution/interop.py` now warns (`RuntimeWarning`, once per process) the first time you touch the Qiskit bridge on `sys.platform == 'darwin'`, but it does not block — some Qiskit/macOS combinations may work fine. If you hit a crash, `pip install dense-evolution[pennylane]` gives the same circuit-interop functionality without constructing any Qiskit object.

</details>

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

# parse any OpenQASM 2.0 / 3.0 string -- single-qubit rotations, a barrier
# (a real OpenQASM synchronization marker: parsed like hardware would, no
# effect on the simulated state), then an entangling layer
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rx(pi/3) q[0];
ry(pi/4) q[1];
h q[2];
barrier q;
cx q[0], q[1];
cx q[1], q[2];
rz(pi/6) q[2];
"""

parser = QASMParser()
circuit = parser.parse(qasm)

sim = DenseSVSimulator(n_qubits=3)
sim.run_circuit_jit(circuit.to_tuples())

probs = sim.get_probabilities()
sv    = sim.get_statevector()
# probs = [0.3201 0.3201 0.0549 0.0549 0.0183 0.0183 0.1067 0.1067]
```

**Noise:**

```python
import numpy as np
from dense_evolution import DenseSVSimulator, QASMParser, NoiseModel

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = QASMParser().parse(qasm)

sim = DenseSVSimulator(n_qubits=2)
sim.run_circuit_jit(circuit.to_tuples())

noisy_sv = NoiseModel.apply_to_sv(np.asarray(sim.sv), n=2, model='depolarizing', p=0.05, rng=np.random.default_rng(0))
np.abs(noisy_sv) ** 2
# [0.   0.5  0.5  0.  ]
```

**VQE:**

```python
import jax
import jax.numpy as jnp
from dense_evolution import QASMParser, circuit_to_energy_fn

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; ry(0.0) q[0];'
circuit = QASMParser().parse(qasm)
energy_fn, n_params = circuit_to_energy_fn(circuit, n_qubits=1)
h = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)  # Pauli Z

theta = jnp.array([0.1])
grad_fn = jax.value_and_grad(energy_fn, argnums=0, has_aux=True)
for _ in range(40):
    (energy, sv), grad = grad_fn(theta, h)
    theta = theta - 0.5 * grad
# energy = -1.0, theta = [3.14159265]
```

**Mitigation (ZNE):**

```python
import dense_evolution as de

e1, e2, e3 = 1.234, 0.876, 0.611  # values at 1x, 2x, 3x noise
de.zero_noise_extrapolation([e1, e2, e3], [1.0, 2.0, 3.0])
# 1.622
```

**Dashboard (local, Streamlit):**

```bash
pip install "dense-evolution[dashboard]"  # JAX already included by default
streamlit run tools/dashboard/app.py
```

**Anti-OOM for large circuits:**

```python
from dense_evolution import Chunk, QASMParser

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[27]; h q[0:27];'
circuit = QASMParser().parse(qasm)

sim = Chunk(27)
sim.run_chunk(circuit.to_tuples(), chunk_size_gates=500)
```

---


## ▍ Key Features

<img src="docs/assets/readme_hero_circuit.png" width="400px" align="right">

- **JIT-fused statevector engine.** No Kronecker-product overhead — stride-sliced linear kernel fusion compiled through JAX XLA, real GPU/TPU dispatch with no code change. [Simulator](https://tatopenn-cell.github.io/Dense-Evolution/api/simulator/) · [MPS backend](https://tatopenn-cell.github.io/Dense-Evolution/api/mps/) for low-entanglement circuits at scale.
- **Anti-OOM `Chunk` engine.** Circuits too large for one array, split dynamically and sized off the real compute device's own free memory — with disk-backed overflow past even that ceiling. [Chunk guide](https://tatopenn-cell.github.io/Dense-Evolution/api/chunk/).
- **Real noise, real mitigation.** Stochastic Kraus channels, real-device noise imported from Qiskit backends, and Zero-Noise Extrapolation to correct for it. [Noise](https://tatopenn-cell.github.io/Dense-Evolution/api/noise/) · [Mitigation](https://tatopenn-cell.github.io/Dense-Evolution/api/mitigation/) · [what noise/mitigation/healing each mean](https://tatopenn-cell.github.io/Dense-Evolution/concepts/).
- **Differentiable VQE, from scratch.** `circuit_to_energy_fn` is the same JAX-differentiable engine real molecular VQE runs on — real Hartree-Fock Hamiltonians, UCCSD/hardware-efficient ansätze, Adam optimization. [Autodiff](https://tatopenn-cell.github.io/Dense-Evolution/api/autodiff/).
- **OpenQASM 2.0/3.0, both directions.** A real parser, plus Qiskit/PennyLane interop bridges. [QASM Parser](https://tatopenn-cell.github.io/Dense-Evolution/api/parser/) · [Interop](https://tatopenn-cell.github.io/Dense-Evolution/api/interop/).
- **Code-agnostic QEC decoding**, Majorana/Jordan-Wigner fermion mapping, from-scratch Hartree-Fock for elements outside PennyLane's own basis set, and a traversable-wormhole-inspired teleportation protocol — see the [full API reference](https://tatopenn-cell.github.io/Dense-Evolution/api/) for all of it.

## ▍ Benchmarks

Measured on Windows, CPU only, 8 GB RAM — PennyLane's `default.qubit` allocates the full statevector; Dense Evolution's `Chunk` holds constant ~2 GB regardless of qubit count.

| Qubits | Hilbert Space | PennyLane | Dense Evolution | Chunk Geometry |
|:------:|:-------------:|:---------:|:----------------:|:--------------:|
| 26 | 67,108,864 | ✅ 1,074 MB | ✅ 2,050 MB | 1× (2²⁷) |
| 28 | 268,435,456 | ❌ OOM | ✅ 2,050 MB | 2× (2²⁷) |
| 32 | 4,294,967,296 | ❌ OOM | ✅ 2,048 MB | 32× (2²⁷) |

On Google Colab (12 GB RAM), n=28 runs in ~3s end-to-end (JIT-compiled, `num_chunks=2`) — the multi-chunk path is fast, not just OOM-safe. Past the real RAM ceiling (`Chunk` alone raises `MemoryPressureError` cleanly rather than crashing the process), `allow_disk_overflow=True` falls back to a slower disk-backed path instead of failing — correctness-first, not benchmarked for speed yet. Full measured table: [docs/api/chunk.md](https://tatopenn-cell.github.io/Dense-Evolution/api/chunk/).


## ▍ Composer & MCP Server

A real circuit editor (graphical or OpenQASM) running on your own machine, plus an MCP server exposing 22 tools so an agent (Claude Code, Claude Desktop, ...) can drive it directly. [Composer](https://tatopenn-cell.github.io/Dense-Evolution/composer/) · [MCP Server](https://tatopenn-cell.github.io/Dense-Evolution/mcp/).

## ▍ Key Resources

- [Full documentation & API reference](https://tatopenn-cell.github.io/Dense-Evolution/)
- [Getting Started guide](https://tatopenn-cell.github.io/Dense-Evolution/getting-started/)
- [Worked examples](https://tatopenn-cell.github.io/Dense-Evolution/examples/)
- [Issue Tracker](https://github.com/tatopenn-cell/Dense-Evolution/issues)

---

## ▍ Changelog

📜 [Full Changelog & Releases](https://github.com/tatopenn-cell/Dense-Evolution/releases) — every version, latest first.

---

## ▍ License

**Business Source License 1.1** — converts automatically to **Apache 2.0** on **1 June 2029**.

- Non-commercial use: unrestricted
- Commercial use: ≤ 24 allocated qubits · ≤ 1,000 circuits/day · ≤ 10,000 shots/circuit
- Attribution required: `© 2026 Salvatore Pennacchio <jtatopenn@libero.it> — Dense Evolution`

Full text: [LICENSE.md](LICENSE.md)

---

## ▍ Cite This

If Dense-Evolution is useful in academic work, please cite it via the metadata in [CITATION.cff](CITATION.cff) (recognized by GitHub's own "Cite this repository" button, and by reference managers that support the [Citation File Format](https://citation-file-format.github.io/)).

Archived on [Zenodo](https://zenodo.org/):

- **Concept DOI** (always resolves to the latest version): [10.5281/zenodo.21855643](https://doi.org/10.5281/zenodo.21855643)
- **This release (v8.1.61)**: [10.5281/zenodo.22009005](https://doi.org/10.5281/zenodo.22009005)

---

<div align="center">
  <sub>© 2026 Salvatore Pennacchio — Dense Evolution</sub>
</div>
