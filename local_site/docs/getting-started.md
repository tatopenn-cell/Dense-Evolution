# Getting Started

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
sim.run_circuit_jit_beast_mode(circuit.to_tuples())

probs = sim.get_probabilities()
sv    = sim.get_statevector()
```

## Anti-OOM for large circuits

```python
from dense_evolution import Chunk

sim = Chunk(27)                                    # logical 27 qubits
circuit_ops = [['h', i] for i in range(27)]
sim.run_chunk(circuit_ops, chunk_size_gates=500)   # SafeMemoryGuard active
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

N_QUBITS, SCALES, K = 2, (1.0, 2.0, 3.0), 200
rng = np.random.default_rng(0)

sim = de.DenseSVSimulator(N_QUBITS)
sim.run_circuit([("h", 0), ("cx", 0, 1)])
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

## Dashboard (local, Streamlit)

`app_dashboard.py` lives at the root of the cloned repository -- it is not part of the
pip-installed package, so this needs the `git clone` from the [Install](#install) section
above, not just `pip install`.

```bash
pip install "dense-evolution[dashboard]"  # JAX already included by default
cd Dense-Evolution
streamlit run app_dashboard.py
```

## Running the test suite

```bash
pip install -e .[dev]
pytest tests/ -v

# with coverage
pytest tests/ --cov=dense_evolution --cov-report=term-missing
```
