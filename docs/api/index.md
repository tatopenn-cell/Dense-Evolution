# API Reference

Each page is a guide to one module: what it does in plain words, a QASM example run
through the real parser, then the reference generated from the package's own docstrings
via [`mkdocstrings`](https://mkdocstrings.github.io/), so the reference can't drift out of
sync with the code.

**Driving these modules without writing Python**: the Composer kernel runs the same
modules from a browser ([Composer](../composer.md)) or from an MCP-aware agent
([MCP Server](../mcp.md), 32 tools — Claude Code, Claude Desktop, ...). Both call the
exact same kernel, not a separate reimplementation.

## Simulators and circuits

| Module | What it's for |
|---|---|
| [Simulator](simulator.md) | Core dense statevector engine (`DenseSVSimulator`) |
| [MPS Simulator](mps.md) | Matrix-product-state engine for low-entanglement circuits at scale |
| [Chunk](chunk.md) | Anti-OOM statevector chunking, including distributed multi-device dispatch |
| [QASM Parser](parser.md) | OpenQASM 2.0 / 3.0 parsing, including user-defined `gate` definitions |
| [Compiler](compiler.md) | Circuit transpilation (`QuantumTranspiler`) |
| [Registry](registry.md) | Hardware detection: how many qubits this machine can safely simulate |
| [Gates](gates.md) | Gate matrix tables (`GATES`, `PARAMETRIC_GATES`, `GATE_IDS`) |
| [Topology](topology.md) | Entangling-layer patterns for variational circuits |
| [States](states.md) | Common state-preparation circuits (GHZ, ...) |
| [QFT](qft.md) | Quantum Fourier Transform circuit |
| [Arithmetic](arithmetic.md) | Register adders, subtractors, comparators, modular add / multiply / power |
| [Postselection](postselect.md) | Keep only the runs where a qubit lands in a chosen state |
| [Random Circuit](random_circuit.md) | Random valid circuits for benchmarking and fuzz-testing |
| [Drawing](drawing.md) | Plain-text circuit diagrams |
| [Diagram](diagram.md) | Quirk-style box diagrams (`plot_circuit`) |
| [Interop](interop.md) | Qiskit / PennyLane bridges |

## Measurement, noise and mitigation

| Module | What it's for |
|---|---|
| [Observables](observables.md) | Pauli-string expectation values and Pauli-sum Hamiltonians |
| [Measurement](measurement.md) | Shot sampling and pure-state fidelity |
| [Entropy](entropy.md) | Multi-qubit partial trace, von Neumann entropy, mutual information |
| [Noise](noise.md) | `NoiseModel` Kraus channels, density-matrix channels, time-varying noise |
| [Mitigation](mitigation.md) | Zero-Noise Extrapolation (scalar, density-matrix, single-qubit Bloch vector), shot allocation, density-matrix diagnostics |
| [Healing](healing.md) | Predictive-healing primitives |
| [QEC](qec.md) | Erasure-aware stabilizer decoding, MWPM, syndrome-based noise estimation |
| [Cryptography Protocols](protocols.md) | BB84, device-independent QKD and conference key agreement (crypto-q) |

## Chemistry and materials

| Module | What it's for |
|---|---|
| [Autodiff](autodiff.md) | Differentiable circuit-to-energy pipeline for VQE |
| [UCCSD](uccsd.md) | Native excitation circuits for the UCCSD ansatz |
| [Trotter](trotter.md) | Real-time Hamiltonian evolution as an actual gate circuit |
| [Fermions](fermions.md) | Majorana-fermion → qubit (Jordan-Wigner) mapping |
| [Native Hartree-Fock](native_hf.md) | From-scratch JAX/Obara-Saika Hartree-Fock, with libcint integrals on Windows, macOS and Linux |
| [QM/MM](qmmm.md) | Region partitioning, embedding, Hellmann-Feynman forces, molecular dynamics |
| [Harrison Tight-Binding](harrison_tb.md) | Universal (materials-independent) sp3 tight-binding Hamiltonians |
| [VHD Tight-Binding](vhd_tb.md) | Material-specific sp3s* tight-binding, validated against real GaAs/Si/Ge gaps |
| [Mass Decomposition](mass_decomposition.md) | Molecular formulas and neutral losses from mass-spectrometry peaks |

## Tools built on the library

These live under `tools/`, not inside the `dense_evolution` package: the vector-healing,
adversarial-attack and retrieval utilities (`ia_utils`) and the Composer kernel
(`dashboard_core`) behind the Composer, the MCP server and the Streamlit Dashboard.

| Module | What it's for |
|---|---|
| [IA Utils — Vector Healing](ia_utils_vector_healing.md) | Repair corrupted steps in a vector sequence (VQE iterations, MD steps, embeddings) |
| [IA Utils — Adversarial Vector Attack](ia_utils_adversarial_vector_attack.md) | Adversarial tests of vector healing's dynamics-versus-noise decision |
| [IA Utils — Hybrid Retrieval (RAG)](ia_utils_rag.md) | Hybrid keyword + embedding passage retrieval over a document collection |
| [Dashboard Core — Engine](dashboard_core_engine.md) | Runs the actual `DenseSVSimulator` circuits for the dashboard |
| [Dashboard Core — Wormhole](dashboard_core_wormhole.md) | SYK traversable-wormhole-inspired teleportation |
| [Dashboard Core — VQE](dashboard_core_vqe.md) | VQE ansatz circuits generated for molecular Hamiltonians |
| [Dashboard Core — Hamiltonians](dashboard_core_hamiltonians.md) | Molecular Hamiltonians built on demand from atomic geometry |
| [Dashboard Core — QM/MM](dashboard_core_qmmm.md) | Backward-compatible re-export of `dense_evolution.qmmm` |
| [Dashboard Core — Mitigation](dashboard_core_mitigation.md) | ZNE panel wired to the engine and its noise channels |
| [Dashboard Core — Vector Healing](dashboard_core_vector_healing.md) | Predictive-healing pass over a noisy vector sequence |
| [Dashboard Core — QASM Library](dashboard_core_qasm_library.md) | Standard OpenQASM 2.0 presets (Bell pair, GHZ, W-state, ...) |
| [Dashboard Core — System Limits](dashboard_core_system_limits.md) | Per-machine qubit limits from available RAM |
| [Dashboard Core — Circuit Diagram](dashboard_core_circuit_diagram.md) | Native matplotlib circuit diagrams |
| [Dashboard Core — State Visuals](dashboard_core_state_visuals.md) | Histogram, Q-sphere, per-qubit Bloch spheres |
| [Dashboard Core — Visuals](dashboard_core_visuals.md) | Native matplotlib visualizations used across the dashboard |
| [Dashboard Core — Graphical Builder](dashboard_core_graphical_builder.md) | Converts drag-and-drop builder ops into gate tuples |
| [Dashboard Core — Circuit Builder Component](dashboard_core_circuit_builder_component.md) | Drag-and-drop circuit builder component |
