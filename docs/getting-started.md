# Getting Started

Run up to 28 qubits in about 3 seconds, JIT-compiled, without crashing -- this page gets you from install to your first circuit in a few minutes.

## Install

```bash
pip install dense-evolution  # JAX is a core dependency, installed by default

# full stack: GPU · dashboard · Qiskit/PennyLane interop
pip install dense-evolution[full]

# just the interop bridge
pip install dense-evolution[qiskit]
pip install dense-evolution[pennylane]

# development (includes pytest + pytest-cov)
git clone https://github.com/tatopenn-cell/Dense-Evolution.git
cd Dense-Evolution && pip install -e .[full,dev]
```

**Google Colab (3 lines):**

```python
!git clone https://github.com/tatopenn-cell/Dense-Evolution.git
%cd Dense-Evolution
!pip install -e .
```

## Quick start

A qubit starts in state `|0>`. `h` (Hadamard) puts it into an equal superposition of
`|0>` and `|1>` -- measuring it afterward gives each outcome with 50% probability.
`cx` (CNOT) entangles two qubits: it flips the second qubit only if the first is `|1>`.
The circuit below puts qubit 0 into superposition, then uses two `cx` gates to spread
that same randomness to qubits 1 and 2 -- all three qubits end up perfectly correlated
(a 3-qubit GHZ state: measuring gives `000` or `111`, each about half the time, never
anything else).

Circuits are always built from OpenQASM text through `QASMParser`, never by
hand-writing the internal gate-tuple format yourself:

```python
from dense_evolution import DenseSVSimulator, QASMParser

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
sim.run_circuit_jit(circuit.to_tuples())

probs = sim.get_probabilities()
sv    = sim.get_statevector()
```

## Anti-OOM for large circuits

```python
from dense_evolution import Chunk, QASMParser

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[27]; h q[0:27];'
circuit = QASMParser().parse(qasm)

sim = Chunk(27)                                    # logical 27 qubits
sim.run_chunk(circuit.to_tuples(), chunk_size_gates=500)   # SafeMemoryGuard active
```

## Zero-Noise Extrapolation

Self-contained: builds its own noisy density matrices via Monte Carlo, so it runs as-is.
`rho_ideal` is used only to grade the result at the end, never fed into the correction
itself (see the full writeup in [Examples](examples.md#density-matrix-zne-healing)).

```python
import numpy as np
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import zne_density_matrix, uhlmann_fidelity
from dense_evolution import QASMParser

N_QUBITS, SCALES, K = 2, (1.0, 2.0, 3.0), 200
rng = np.random.default_rng(0)

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = QASMParser().parse(qasm)

sim = de.DenseSVSimulator(N_QUBITS)
sim.run_circuit_jit(circuit.to_tuples())
ideal_sv = np.asarray(sim.get_statevector())
rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

def noisy_density_matrix(p):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(K):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, 'depolarizing', p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    return jnp.asarray(rho / K, dtype=jnp.complex128)

rho_at_scales = jnp.stack([noisy_density_matrix(0.05 * scale) for scale in SCALES])

raw_fidelity = uhlmann_fidelity(rho_at_scales[0], rho_ideal)
corrected = zne_density_matrix(rho_at_scales, SCALES)
corrected_fidelity = uhlmann_fidelity(corrected, rho_ideal)
```

See [`dense_evolution.mitigation`](api/mitigation.md) for the full API, including the
`_jit` variants for use inside a larger `jax.jit`-compiled pipeline, and
[Examples](examples.md) for this walkthrough plus MPS and differentiable-VQE examples.
See [Benchmarks](benchmarks.md) for how the Anti-OOM Chunk engine above compares to PennyLane.

## Dashboard (local, Streamlit)

`tools/dashboard/app.py` lives in the cloned repository -- it is not part of the
pip-installed package, so this needs the `git clone` from the [Install](#install) section
above, not just `pip install`. Circuit builder / statevector / probabilities / Q-sphere
only -- VQE, molecular Hamiltonians, ZNE mitigation, and vector healing aren't wired into
this Streamlit UI (yet); reach them via [Composer](composer.md) or the [MCP Server](mcp.md)
above instead.

```bash
pip install "dense-evolution[dashboard]"  # JAX already included by default
cd Dense-Evolution
streamlit run tools/dashboard/app.py
```

## MCP Server (drive it from an agent)

Same kernel as the Composer web page above, driven by an MCP-aware agent (Claude Code,
Claude Desktop, ...) instead of a browser -- circuits, molecular energies, VQE, QM/MM
forces, MD trajectories, ZNE mitigation, the wormhole teleportation protocol, and healing a
noisy vector sequence, all as callable tools instead of Python you write yourself.

```bash
pip install "dense-evolution[mcp]"

dense-evolution serve   # start the kernel first, in one terminal
dense-evolution mcp     # in another terminal, or registered with your MCP client's config
```

**Register with Claude Code:**

```bash
claude mcp add dense_evolution -- dense-evolution mcp
```

Full tool reference, setup details, and design notes: **[MCP Server](mcp.md)**.

## Running the test suite

```bash
pip install -e .[dev]
pytest tests/ -v

# with coverage
pytest tests/ --cov=dense_evolution --cov-report=term-missing
```
