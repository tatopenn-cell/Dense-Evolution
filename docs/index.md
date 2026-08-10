# Dense-Evolution

**Dense statevector quantum simulator · JAX XLA · NISQ · VQE · QML**

[![CI](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/tatopenn-cell/Dense-Evolution/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dense-evolution?style=flat-square&color=00e5ff)](https://pypi.org/project/dense-evolution/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/Backend-JAX_XLA-f9ab00?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)

Dense-Evolution is a high-performance statevector simulator for deep NISQ circuits, VQE
pipelines, and QML workloads. It eliminates Kronecker-product overhead via stride-sliced
linear kernel fusion compiled through JAX XLA, keeping memory at the theoretical minimum
of `2ⁿ × 16 bytes`.

Using this in academic work? See [CITATION.cff](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/CITATION.cff) in the repository root for citation metadata. Archived on Zenodo — concept DOI [10.5281/zenodo.21855643](https://doi.org/10.5281/zenodo.21855643).

## What's in here

- **[`DenseSVSimulator`](api/simulator.md)** — the core dense statevector engine, JIT-fused
  circuit execution, projective measurement.
- **[`MPSSimulator`](api/mps.md)** — matrix-product-state engine for low-entanglement
  circuits at hundreds of qubits, with a `jax.lax.scan`-fused execution path.
- **[`Chunk`](api/chunk.md)** — anti-OOM statevector chunking for circuits too large for a
  single dense allocation, including a real multi-device distributed dispatch path.
- **[`QASMParser`](api/parser.md)** — OpenQASM 2.0 / 3.0 parsing, including `for` loops,
  range syntax, and QASM3 classical `bit` registers.
- **[`dense_evolution.mitigation`](api/mitigation.md)** — Zero-Noise Extrapolation, both the
  classic scalar/vector form and a density-matrix extension (Smolin-Gambetta-Smith physical
  projection, Uhlmann fidelity), every entry point with a `jax.jit`-compatible variant.
- **[`dense_evolution.healing`](api/healing.md)** — the predictive-healing primitives
  (`calculate_delta_preemp`, `calculate_vettore_dinamico`, ...) ZNE's healing-adapted branch
  is built on, and that `dashboard_core.run_vector_healing` / `dense_evolution_vector_healing`
  (below) apply to real noisy vector sequences.
- **[`ia_utils.vector_healing`](api/ia_utils_vector_healing.md)** — `median_healing` /
  `enhanced_dense_healing_hybrid`, applying the Phi-Trigger primitives above to real
  `(n_steps, dim)` vector sequences (VQE/MD telemetry, embeddings), NaN/Inf-safe.
- **[`ia_utils.adversarial_vector_attack`](api/ia_utils_adversarial_vector_attack.md)** — a
  gradient-based (PGD-style) robustness test that crafts the minimal perturbation flipping
  `enhanced_dense_healing_hybrid`'s Phi-Trigger decision either direction.
- **[`NoiseModel`](api/registry.md)** — Kraus-channel noise (depolarizing, bitflip,
  phaseflip, amplitude damping, and a combined worst-case model), JAX- and NumPy-native.
- **Interop** with [Qiskit and PennyLane](api/interop.md), and [autodiff](api/autodiff.md)
  through `circuit_to_energy_fn` for gradient-based VQE.

## Dashboard Core — Composer's real compute layer

`dashboard_core` (a separate top-level package, `pip install`ed alongside `dense_evolution`
and `ia_utils`) is what Composer and the MCP server both actually call underneath — real
physics and real rendering, not mocked results glued to a UI:

**Science**: [Wormhole (SYK teleportation)](api/dashboard_core_wormhole.md) ·
[VQE](api/dashboard_core_vqe.md) · [Hamiltonians](api/dashboard_core_hamiltonians.md) ·
[QM/MM](api/dashboard_core_qmmm.md)

**Mitigation & healing**: [Mitigation (ZNE panel)](api/dashboard_core_mitigation.md) ·
[Vector Healing (healing panel)](api/dashboard_core_vector_healing.md)

**Infrastructure**: [Engine](api/dashboard_core_engine.md) ·
[System Limits](api/dashboard_core_system_limits.md) ·
[QASM Library](api/dashboard_core_qasm_library.md)

**Visualization & UI**: [Circuit Diagram](api/dashboard_core_circuit_diagram.md) ·
[State Visuals](api/dashboard_core_state_visuals.md) ·
[Visuals](api/dashboard_core_visuals.md) ·
[Graphical Builder](api/dashboard_core_graphical_builder.md) ·
[Circuit Builder Component](api/dashboard_core_circuit_builder_component.md)

## Beyond the library: Composer and MCP

Two ways to drive the simulator without writing Python against it directly, both backed by
the same local kernel (`local_site/app/server.py`, real `DenseSVSimulator`/PennyLane
Hartree-Fock, no mocked physics):

- **[Composer](composer.md)** — a browser UI: build a circuit graphically or in OpenQASM,
  compute molecular ground-state energies, run VQE, get QM/MM forces and MD trajectories,
  apply ZNE mitigation, and run the [traversable-wormhole-inspired teleportation
  protocol](api/dashboard_core_wormhole.md).
- **[MCP Server](mcp.md)** — the same kernel, driven by an MCP-aware agent (Claude Code,
  Claude Desktop, ...) instead of a browser: 21 tools covering everything Composer does,
  plus a batch energy scan, a batch wormhole sweep, and healing a noisy vector sequence
  (`dense_evolution_vector_healing`, see `dense_evolution.healing` above).

## Module dependencies

Real module graph, traced from the actual `import` statements in each file (not an
idealized layering) -- circuit sources feed three interchangeable execution engines, which
in turn feed the higher-level mitigation/autodiff layers.

```mermaid
flowchart LR
    subgraph core["Core"]
        parser["parser<br/>(QASMParser)"]
        gates["gates"]
        registry["registry<br/>(NoiseModel)"]
        compiler["compiler<br/>(QuantumTranspiler)"]
    end

    subgraph engines["Engines"]
        simulator["simulator<br/>(DenseSVSimulator)"]
        mps["mps<br/>(MPSSimulator)"]
        chunk["chunk<br/>(Chunk)"]
    end

    subgraph higher["Higher-level"]
        interop["interop<br/>(Qiskit / PennyLane)"]
        autodiff["autodiff<br/>(circuit_to_energy_fn)"]
        healing["healing<br/>(predictive primitives)"]
        mitigation["mitigation<br/>(ZNE)"]
    end

    subgraph iautils["ia_utils (separate top-level package)"]
        vhealing["vector_healing<br/>(median_healing, enhanced_dense_healing_hybrid)"]
        advattack["adversarial_vector_attack<br/>(craft_adversarial_healing_perturbation)"]
    end

    subgraph dashcore["dashboard_core (separate top-level package, Composer's backend)"]
        dc_wormhole["wormhole"]
        dc_vqe["vqe"]
        dc_hamiltonians["hamiltonians"]
        dc_qmmm["qmmm"]
        dc_mitigation["mitigation<br/>(ZNE panel)"]
        dc_vector_healing["vector_healing<br/>(healing panel)"]
        dc_engine["engine"]
        dc_qasm_library["qasm_library"]
        dc_system_limits["system_limits"]
        dc_circuit_diagram["circuit_diagram"]
        dc_state_visuals["state_visuals"]
        dc_visuals["visuals"]
        dc_graphical_builder["graphical_builder"]
        dc_circuit_builder_component["circuit_builder_component"]
    end

    gates --> registry
    simulator --> registry
    simulator --> gates
    simulator --> compiler
    mps --> compiler
    mps --> gates
    chunk --> simulator
    chunk --> compiler
    chunk --> gates
    interop --> parser
    interop --> simulator
    autodiff --> parser
    autodiff --> gates
    autodiff --> compiler
    autodiff --> registry
    mitigation --> healing
    vhealing -.->|lazy import| healing
    advattack --> healing

    dc_vqe --> dc_hamiltonians
    dc_qmmm --> dc_hamiltonians
    dc_visuals --> dc_circuit_diagram
    dc_visuals --> dc_state_visuals
    dc_circuit_builder_component --> dc_graphical_builder
    dc_vector_healing -.->|lazy import| vhealing
```

`dashboard_core.engine`, `dashboard_core.hamiltonians`, `dashboard_core.mitigation`,
`dashboard_core.qasm_library`, `dashboard_core.system_limits`, and `dashboard_core.wormhole`
each depend only on `dense_evolution`'s top-level public API (`import dense_evolution as de`,
or `from dense_evolution import mutual_information, majorana_pauli_terms, trotter_evolve_ops`
for `wormhole`), not on a specific submodule shown above, so they have no internal edges
drawn here beyond that. `circuit_diagram`, `state_visuals`, and `graphical_builder` have no
internal-repo dependencies at all (NumPy / Matplotlib only).

## Where to start

New to the package? Start with **[Getting Started](getting-started.md)**.

Looking for a specific function or class? Jump straight to the **[API
Reference](api/index.md)**.

Want to see what changed recently? Check the **[Changelog](changelog.md)**.

## Honesty as a design principle

This library's docstrings and changelog document real, measured findings — including
negative results and rejected approaches — not just the features that worked. Where a
number is quoted (a speedup, a fidelity improvement, a coverage percentage), it was
measured directly and the measurement is described, not assumed. See the
[Changelog](changelog.md) for the full history.
