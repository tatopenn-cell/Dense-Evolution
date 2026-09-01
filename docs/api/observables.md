# Observables (Pauli-string expectation values)

A Hamiltonian for a real molecule or spin model is almost never handed to you as one
big matrix -- it comes as a weighted sum of Pauli strings, `H = sum_i coeff_i * P_i`
(e.g. `1.0 * ZZ + 0.5 * X0`). This module works with that sum directly: expectation
values, matrix-vector products, and even the full dense matrix when one is genuinely
needed, all without requiring a full `2**n x 2**n` matrix to exist first unless you ask
for one.

## Step 1. Expectation value of one Pauli string

```python
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(circuit.to_tuples())
sv = sim.get_statevector()

de.pauli_expectation(sv, 'ZZ')
```

```
0.9999999999999998
```

`sv` is the Bell state from the [Simulator](simulator.md) page. `pauli_expectation`
reads `'ZZ'` left to right as qubit 0, qubit 1 -- `Z` on both qubits gives `+1` on
`|00>` and on `|11>` (the only two basis states a Bell state has any amplitude on), so
the expectation value is `1`, up to floating-point rounding. A dict form
(`{0: 'Z', 1: 'Z'}`) works identically -- convenient when only a few qubits are
non-identity.

## Step 2. A weighted sum of several Pauli strings

```python
terms = [(1.0, 'ZZ'), (0.5, {0: 'Z'})]
de.pauli_sum_expectation(sv, terms)
```

```
1.4999999999999998
```

`terms` is a list of `(coeff, pauli_string)` pairs -- this is the Hamiltonian
representation the rest of this module (and [`fermions`](fermions.md),
[`hamiltonians.md`](dashboard_core_hamiltonians.md)) build. `pauli_sum_expectation`
is `pauli_expectation` applied to each term and summed with its coefficient: `1.0*<ZZ>
+ 0.5*<Z0> = 1.0*1.0 + 0.5*1.0 = 1.5`, matching the printed value.

## Step 3. The same Hamiltonian, as a dense matrix

```python
import numpy as np

H = de.pauli_hamiltonian_to_matrix(terms, n_qubits=2)
np.real(np.conj(sv) @ H @ sv)
```

```
1.4999999999999998
```

`pauli_hamiltonian_to_matrix` builds the real `(4, 4)` Hermitian matrix for the same
`terms` -- `<psi|H|psi>` computed this way agrees with Step 2's `pauli_sum_expectation`
exactly. Use this when something downstream genuinely needs the explicit matrix (exact
diagonalization for a ground-state energy, say); everything else on this page avoids
building it.

## Step 4. `H @ vector`, without the matrix

```python
from dense_evolution.physics.observables import pauli_sum_matvec

Hv = pauli_sum_matvec(sv, terms, n_qubits=2)
np.allclose(H @ sv, Hv)
```

```
True
```

`pauli_sum_matvec` computes the same `H @ vector` Step 3's dense `H` would, in
`O(dim * n_terms)` instead of ever materializing the `(2**n, 2**n)` matrix -- the
matrix-free primitive behind an iterative eigensolver (`scipy.sparse.linalg.eigsh` via
a `LinearOperator` wrapping this function) for a system too large to diagonalize
densely.

## Step 5. Differentiable: the same Hamiltonian inside `jax.grad`

```python
import jax
import jax.numpy as jnp

qasm_rx = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; rx(0.0) q[0]; cx q[0],q[1];'
circuit_rx = de.QASMParser().parse(qasm_rx)
energy_fn, n_params = de.circuit_to_energy_fn(circuit_rx, n_qubits=2)

h_op = de.PauliSumOperator(terms, n_qubits=2)

def loss(theta):
    energy, sv = energy_fn(theta, h_op)
    return energy

theta = jnp.array([0.8])
jax.value_and_grad(loss)(theta)
```

```
(Array(1.34835335, dtype=float64), Array([-0.35867805], dtype=float64))
```

`pauli_sum_matvec` (Step 4) is `numpy`-based -- fine called directly, but it breaks
under `jax.grad` tracing. `PauliSumOperator` wraps the same `terms` behind `__matmul__`,
backed by a pure-`jnp` rewrite (`pauli_sum_matvec_jax`), so it drops straight into
[`circuit_to_energy_fn`](autodiff.md)'s `h_matrix` argument (the only thing `energy_fn`
does with `h_matrix` is `h_matrix @ statevector`) and stays differentiable end to end.
This is what a real VQE gradient step looks like -- `jax.grad` here needs no dense
Hamiltonian at any point, which matters once a system is too large for one to fit in
memory at all (`pauli_hamiltonian_to_matrix`'s `(2**n, 2**n)` matrix is already 4GB at
14 qubits; `PauliSumOperator` is unaffected).

---

## Details

**Indexing convention**: qubit 0 is the *most* significant bit of the basis-state
index throughout this module (`pauli_terms[0]` in a string is qubit 0) -- matches
[`DenseSVSimulator`](simulator.md) and [`entropy`](entropy.md), not the little-endian
convention some other libraries use.

**`multiply_pauli_terms`** multiplies several Pauli-string *operators* together (order
matters -- unlike every function above, which sums independent terms), tracking the
`i^k` phase from same-qubit collisions (`X*Y = iZ`, etc.) -- the manual Pauli-algebra
primitive behind [`total_parity_operator`](fermions.md)'s Klein-factor construction, or
any other by-hand Pauli-operator product.

**Precision**: `pauli_sum_matvec_jax`/`pauli_sum_expectation_jax`/`PauliSumOperator`
never call `dense_evolution.set_precision(True)` themselves -- if nothing else in the
process has constructed a `DenseSVSimulator`/`circuit_to_energy_fn` yet, JAX is still at
its `float32` default, and Step 5's numbers above would come out at `~1e-7` relative
precision instead of `~1e-16`. Call `de.set_precision(True)` yourself first if using
`PauliSumOperator` standalone, before anything else has a chance to enable `x64` for
you.

::: dense_evolution.physics.observables

---

**See also**: [`circuit_to_energy_fn`](autodiff.md) for the full differentiable-VQE
picture Step 5 is part of; [`fermions`](fermions.md) for Pauli terms built from
Majorana/Jordan-Wigner-mapped fermionic operators instead of by hand.
