# Examples

Four runnable, end-to-end examples, each lifted from real experiments/tests already in
the repository rather than written fresh for this page:

- [Density-matrix ZNE healing](#density-matrix-zne-healing) — from
  [`experiments/matrix_healing_zne.py`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/experiments/matrix_healing_zne.py),
  the exact script `zne_density_matrix`'s docstring reports measured numbers from.
- [MPS for low-entanglement circuits](#mps-for-low-entanglement-circuits) — from
  `tests/test_mps.py::test_run_circuit_jit_ghz_chain`.
- [Differentiable VQE](#differentiable-vqe) — from
  `tests/test_autodiff.py::TestCircuitToEnergyFn`.
- [Vector healing](#vector-healing) — from `tests/test_ia_healing.py`, healing a noisy
  vector sequence (not to be confused with density-matrix ZNE healing above).

## Density-matrix ZNE healing

Zero-noise extrapolation on full density matrices: run a circuit at several *scaled* noise
levels, extrapolate back to the zero-noise limit, then project the (generally unphysical)
extrapolated result back onto the physical cone of valid density matrices
([Smolin-Gambetta-Smith 2012](https://arxiv.org/abs/1106.5458)).

The ideal state (`rho_ideal`) is built once here purely to *grade* the result at the end.
It is never fed into the noise ensemble, the extrapolation, or the physical-projection
step — feeding it back in anywhere but the final fidelity check would be oracle access,
not error mitigation.

```python
import numpy as np
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix

N_QUBITS = 2
BASE_P = 0.05
SCALES = (1.0, 2.0, 3.0)
K_TRAJECTORIES = 200


def bell_state_sv():
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    return np.asarray(sim.get_statevector())


def noisy_density_matrix(ideal_sv, p, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, 'depolarizing', p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


ideal_sv = bell_state_sv()
rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

rng = np.random.default_rng(0)
rho_at_scales = jnp.stack([
    noisy_density_matrix(ideal_sv, BASE_P * scale, K_TRAJECTORIES, rng)
    for scale in SCALES
])

raw_fidelity = uhlmann_fidelity(rho_at_scales[0], rho_ideal)          # base-scale, uncorrected
corrected = zne_density_matrix(rho_at_scales, SCALES)                 # Richardson + physical projection
corrected_fidelity = uhlmann_fidelity(corrected, rho_ideal)           # grading only

print(f"raw fidelity:       {raw_fidelity:.4f}")
print(f"corrected fidelity: {corrected_fidelity:.4f}")
```

Measured on this exact script (5 seeds, 400 trajectories each, 2–5 qubits, 5 noise
channels): 96/100 runs improve fidelity, mean delta +0.12. See the
[Changelog](changelog.md) and [`dense_evolution.mitigation`](api/mitigation.md) docstrings
for the full sweep and the honest negative results (predictive-healing coefficient
perturbation was tested and rejected for the matrix case — negligible effect even
amplified 100x).

Run it yourself:

```bash
python experiments/matrix_healing_zne.py
```

## MPS for low-entanglement circuits

`MPSSimulator` keeps a bounded bond dimension (`max_bond`), trading exactness for circuits
that stay low-entanglement (GHZ chains, shallow QAOA layers, most NISQ ansätze) at qubit
counts a dense statevector could never hold. `run_circuit_jit` fuses the whole circuit into
a single `jax.lax.scan`.

```python
import numpy as np
from dense_evolution import MPSSimulator

n = 6
ops = [["h", 0]] + [["cx", q, q + 1] for q in range(n - 1)]  # GHZ chain

mps = MPSSimulator(n_qubits=n, max_bond=8)
mps.run_circuit_jit(ops)

prob = np.abs(np.asarray(mps.contract_to_statevector())) ** 2
print(prob[0], prob[2 ** n - 1])  # both ~0.5, everything else ~0 -- the GHZ signature
```

See [`dense_evolution.mps`](api/mps.md) for the truncated-SVD mechanics and the
`jax.lax.scan` fusion this relies on.

## Differentiable VQE

`circuit_to_energy_fn` turns a parsed circuit into a `(theta, hamiltonian) -> (energy,
statevector)` function that is differentiable end-to-end with `jax.grad`/`jax.value_and_grad`
— no manual parameter-shift rule needed. This runs a short Adam loop against a random
diagonal Hamiltonian.

```python
import numpy as np
import jax
import jax.numpy as jnp
import dense_evolution as de

VQE_QASM = (
    'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    'ry(0.5) q[0]; rx(0.5) q[1]; cx q[0],q[1]; rz(0.2) q[1]; cx q[0],q[1]; '
    'ry(0.5) q[0]; rx(0.5) q[1]; measure q -> c;'
)


def random_hamiltonian(n_qubits, seed=7):
    rng = np.random.default_rng(seed)
    values = np.sort(rng.uniform(-2.5, 2.5, 2 ** n_qubits))
    return jnp.diag(jnp.array(values, dtype=jnp.float64))


circ = de.QASMParser().parse(VQE_QASM)
energy_fn, n_params = de.circuit_to_energy_fn(circ, circ.n_qubits)
h_matrix = random_hamiltonian(circ.n_qubits)

rng = np.random.default_rng(3)
theta = rng.uniform(-np.pi, np.pi, n_params)

energy_and_grad = jax.jit(jax.value_and_grad(energy_fn, argnums=0, has_aux=True))
m, v = np.zeros(n_params), np.zeros(n_params)
lr, beta1, beta2 = 0.1, 0.9, 0.999

for epoch in range(1, 31):
    (energy, _), grad = energy_and_grad(jnp.asarray(theta), h_matrix)
    grad = np.asarray(grad)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat, v_hat = m / (1 - beta1 ** epoch), v / (1 - beta2 ** epoch)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

print(f"final energy:  {float(energy):.4f}")
print(f"ground state:  {float(jnp.min(jnp.diag(h_matrix))):.4f}")
```

`circuit_to_energy_fn` also accepts an optional `noise=` (`dense_evolution.NoiseSpec`, a
JAX pytree) to trace noisy VQE runs natively — composable with `jax.jit`, `jax.grad`, and
`jax.vmap` over noise-key batches. See [`dense_evolution.autodiff`](api/autodiff.md).

## Vector healing

Not to be confused with [density-matrix ZNE healing](#density-matrix-zne-healing) above —
same word, different thing. This is `dense_evolution.healing`'s Phi-Trigger applied to a
real `(n_steps, dim)` vector sequence (a VQE parameter/energy trajectory, MD telemetry, or
any other noisy sequence): per step, keep it if the change from a local baseline looks like
genuine dynamics, replace it with the local median if it looks like static noise. NaN/Inf
entries are always sanitized first, regardless of that decision.

```python
import numpy as np
from ia_utils.vector_healing import enhanced_dense_healing_hybrid

rng = np.random.default_rng(0)
trajectory = rng.normal(loc=1.0, scale=0.05, size=(30, 4))  # e.g. a VQE parameter history
trajectory[17] = [50.0, -30.0, 12.0, -8.0]                  # one genuine outlier step

healed, meta = enhanced_dense_healing_hybrid(trajectory)

print(f"outlier step replaced: {not np.array_equal(healed[17], trajectory[17])}")
print(f"fallback_triggered:    {meta['fallback_triggered']}")     # NaN/Inf found *and* corrected
print(f"reconstruction_error:  {meta['reconstruction_error']:.4f}")
```

Same primitives, reachable without writing Python: `dashboard_core.run_vector_healing`
(kernel-facing wrapper), the Composer kernel's `POST /api/vector_healing`, or the MCP tool
`dense_evolution_vector_healing` — see [MCP Server](mcp.md).
