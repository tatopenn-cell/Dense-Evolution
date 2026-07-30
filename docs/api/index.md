# API Reference

Generated directly from the package's own docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/) — nothing here is duplicated by hand, so
it can't drift out of sync with the code.

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
