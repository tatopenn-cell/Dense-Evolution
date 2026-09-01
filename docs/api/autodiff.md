# Autodiff (differentiable VQE)

VQE (Variational Quantum Eigensolver) needs two things at every optimization step: the
current energy, and the gradient of that energy with respect to the circuit's
parameters, so an optimizer knows which way to adjust them. `circuit_to_energy_fn`
turns a parsed circuit into a plain JAX function that computes both -- differentiable
with `jax.grad`, the same way you'd differentiate any other JAX function, no manual
gradient formula required.

## Step 1. From a circuit to an energy function

```python
import dense_evolution as de
from dense_evolution.physics.observables import pauli_hamiltonian_to_matrix

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; ry(0.0) q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
energy_fn, n_params = de.circuit_to_energy_fn(circuit, n_qubits=2)

H = pauli_hamiltonian_to_matrix([(1.0, 'ZZ'), (0.5, {0: 'Z'})], n_qubits=2)
energy, sv = energy_fn([0.3], H)
energy
```

```
1.4776682445628029
```

`circuit_to_energy_fn(circuit, n_qubits)` returns `energy_fn` and `n_params` --
`ry(0.0)`'s literal `0.0` is only a placeholder (any parametric gate's written value is
ignored and injected from `theta` instead, in the order those gates appear), so
`n_params=1` here, matching the circuit's one `ry`. Calling `energy_fn(theta, H)` runs
the circuit with `theta` in place of that placeholder, then returns
`(<psi|H|psi>, psi)` -- the energy and the resulting statevector.

## Step 2. The gradient, for real

```python
import jax
import jax.numpy as jnp

def loss(theta):
    energy, sv = energy_fn(theta, H)
    return energy

theta = jnp.array([0.3])
jax.value_and_grad(loss)(theta)
```

```
(Array(1.47766824, dtype=float64), Array([-0.1477601], dtype=float64))
```

`energy_fn` is a pure JAX function, so `jax.grad`/`jax.value_and_grad` work on it
directly -- no parameter-shift rule, no finite differences, no separate gradient
circuit to build. This is the actual VQE gradient step: feed `grad` to any JAX
optimizer (`optax.adam`, for instance) and repeat to find the ansatz parameters that
minimize the energy.

## Step 3. Noise, inside the same traced call

```python
noise = de.NoiseSpec(model='depolarizing', p=0.1, jax_key=jax.random.PRNGKey(0))
energy_noisy, sv_noisy = energy_fn(theta, H, noise=noise)
energy_noisy
```

```
-0.522331755437196
```

`noise`, when given, is a [`NoiseSpec`](noise.md) applied to the statevector right
after the circuit runs and before the energy is computed -- inside the same traced
computation as `theta`, not an external step spliced in around `energy_fn`. Because
`NoiseSpec` carries its own `jax_key` as a JAX pytree leaf, this whole call stays
`jit`/`grad`/`vmap`-composable with no Python-side random-number bookkeeping needed.

---

## Details

**Precision**: `circuit_to_energy_fn` is one of the entry points that lazily enables
`jax_enable_x64` the first time it's called -- Step 2's gradient above comes out as
`float64` for that reason, not JAX's `float32` default.

**Past ~14 qubits, `H` shouldn't be this dense matrix**: `energy_fn`'s only use of `H`
is `H @ statevector` -- a `(2**n, 2**n)` matrix is already 4GB at 14 qubits, the
practical ceiling on a typical laptop. [`PauliSumOperator`](observables.md) wraps a
Pauli-sum Hamiltonian (`[(1.0, 'ZZ'), (0.5, {0: 'Z'})]`, the same `terms` format Step
1's `H` was built from) behind `__matmul__`, and drops in as `H` above unchanged --
`energy_fn(theta, PauliSumOperator(terms, n_qubits))` -- without ever building the
dense matrix. See [Observables](observables.md)'s own differentiable-VQE step for the
full worked comparison against this page's dense-matrix path, and
[Dense-Evolution-Discovery's VQE + ZNE + autodiff example](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/vqe_pauli_sum_zne_autodiff/)
for a real 12-qubit molecule built this way.

**Circuits from Qiskit/PennyLane**: circuits imported via `from_qiskit`/`from_pennylane`
([Interop](interop.md)) are not differentiable on their own -- pass them through
`circuit_to_energy_fn` the same way as a `QASMParser`-parsed circuit to get a
`jax.grad`-ready `energy_fn`.

**Unsupported gates fail loudly, not silently**: a gate with no `GATE_IDS` entry (or a
multi-parameter gate like `u2`/`u3`, which this function's one-parameter-per-row
internal template can't represent) raises `ValueError` naming the gate, rather than
being silently dropped from the traced circuit -- decompose it into `rx`/`ry`/`rz`/`cx`
first, or use `DenseSVSimulator.run_circuit` (the eager path) directly if it must stay
as-is.

::: dense_evolution.solvers.autodiff

---

**See also**: [Observables](observables.md) for `PauliSumOperator` and the Pauli-sum
Hamiltonian format this page's `H` argument accepts either as a dense matrix or
matrix-free; [Mitigation](mitigation.md) for Zero-Noise Extrapolation, the standard
next step after Step 3's single noisy sample.
