# API Reference

Generated directly from the package's own docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/) — nothing here is duplicated by hand, so
it can't drift out of sync with the code. Scoped to the `dense_evolution` package itself;
`dashboard_core`, `ia_utils`, and `mcp_server` (the Composer kernel and its MCP adapter) are
built on top of these modules but aren't part of this generated reference — see below.

**Driving these modules without writing Python**: every module in the table has a real,
tested execution path through the Composer kernel, reachable either from a browser
([Composer](../composer.md)) or from an MCP-aware agent ([MCP Server](../mcp.md), 21 tools —
Claude Code, Claude Desktop, ...). Both call the exact same kernel, not a separate
reimplementation.

| Module | What it's for |
|---|---|
| [Simulator](simulator.md) | Core dense statevector engine (`DenseSVSimulator`) |
| [MPS Simulator](mps.md) | Matrix-product-state engine for low-entanglement circuits at scale |
| [Chunk](chunk.md) | Anti-OOM statevector chunking, including distributed multi-device dispatch |
| [QASM Parser](parser.md) | OpenQASM 2.0 / 3.0 parsing |
| [Compiler](compiler.md) | Circuit transpilation (`QuantumTranspiler`) |
| [Registry & Noise Models](registry.md) | Hardware detection, `NoiseModel` Kraus channels |
| [Gates](gates.md) | Gate matrix tables (`GATES`, `PARAMETRIC_GATES`, `GATE_IDS`) |
| [Mitigation](mitigation.md) | Zero-Noise Extrapolation, scalar and density-matrix |
| [Healing](healing.md) | Predictive-healing primitives |
| [Interop](interop.md) | Qiskit / PennyLane bridges |
| [Autodiff](autodiff.md) | Differentiable circuit-to-energy pipeline for VQE |
| [Harrison Tight-Binding](harrison_tb.md) | Universal (materials-independent) sp3 tight-binding Hamiltonians |
| [VHD Tight-Binding](vhd_tb.md) | Material-specific sp3s* tight-binding, validated against real GaAs/Si/Ge gaps |
| [Fermions](fermions.md) | Majorana-fermion → qubit (Jordan-Wigner) mapping |
| [Entropy](entropy.md) | Multi-qubit partial trace, von Neumann entropy, mutual information |
| [Trotter](trotter.md) | Real-time Hamiltonian evolution as an actual gate circuit |
| [Native Hartree-Fock](native_hf.md) | From-scratch JAX/Obara-Saika ab-initio HF engine for elements outside PennyLane's own STO-3G table |
