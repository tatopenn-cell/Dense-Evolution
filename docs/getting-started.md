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

```python
import jax.numpy as jnp
from dense_evolution import zne_density_matrix, uhlmann_fidelity

# rho_at_scales[i]: a noisy density-matrix estimate at noise_factors[i]
corrected = zne_density_matrix(rho_at_scales, noise_factors=[1.0, 2.0, 3.0])

# grade against a known ideal state -- never feed this back into correction
fidelity = uhlmann_fidelity(corrected, rho_ideal)
```

See [`dense_evolution.mitigation`](api/mitigation.md) for the full API, including the
`_jit` variants for use inside a larger `jax.jit`-compiled pipeline.

## Dashboard (local, Streamlit)

```bash
pip install "dense-evolution[dashboard]"  # JAX already included by default
streamlit run app_dashboard.py
```

## Running the test suite

```bash
pip install -e .[dev]
pytest test_dense_evolution.py test_mitigation.py test_mps.py -v

# with coverage
pytest --cov=dense_evolution --cov-report=term-missing
```
