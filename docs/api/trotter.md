# Trotter (real-time Hamiltonian evolution as gates)

`exp(-i*H*t)` -- how a quantum state evolves under a Hamiltonian for a real amount of
time -- isn't itself a quantum gate. Trotterization is the standard recipe for turning
it into one anyway: split `H` into pieces a real device already knows how to run
(Pauli-string rotations), and apply them in short, repeated slices. The more slices,
the closer the result gets to the true evolution -- this module builds those slices as
an actual gate circuit, and two related functions for time-dependent evolution that
skips the circuit representation entirely.

## Step 1. One Pauli rotation, exact

```python
import numpy as np
import dense_evolution as de
from dense_evolution.circuits.trotter import pauli_rotation_ops

ops = pauli_rotation_ops({0: 'Z', 1: 'Z'}, 0.6)
ops
```

```
[('cx', 0, 1), ('rz', 1, 1.2), ('cx', 0, 1)]
```

`pauli_rotation_ops(pauli_dict, angle)` builds `exp(-i*angle/2 * P)` as a real gate
sequence -- here `P = Z0 Z1`, the standard "CX-RZ-CX" pattern for a two-qubit `ZZ`
rotation. This is exact, not an approximation: running these three gates on a fresh
`DenseSVSimulator` reproduces `scipy.linalg.expm(-1j * 0.6/2 * ZZ) @ psi0` to fidelity
`1.0` (verified for 1-4-qubit mixed X/Y/Z strings, not just Z-strings).

## Step 2. Many terms, many slices: Trotterization

```python
from dense_evolution.circuits.trotter import trotter_evolve_ops
from dense_evolution.physics.observables import pauli_hamiltonian_to_matrix
from scipy.linalg import expm

terms = [(1.0, {0: 'Z', 1: 'Z'}), (0.5, {0: 'X'}), (0.5, {1: 'X'})]
H = pauli_hamiltonian_to_matrix(terms, n_qubits=2)
psi0 = np.zeros(4, dtype=complex)
psi0[0] = 1.0
exact = expm(-1j * H * 1.0) @ psi0

for n_steps in (1, 5, 20):
    ops = trotter_evolve_ops(terms, t=1.0, n_steps=n_steps)
    sim = de.DenseSVSimulator(2)
    sim.run_circuit_jit(ops)
    sv = sim.get_statevector()
    print(n_steps, abs(np.vdot(sv, exact)) ** 2)
```

```
1 0.7388092965366081
5 0.9923283507781123
20 0.9995352691210072
```

`trotter_evolve_ops(terms, t, n_steps)` applies Step 1's exact single-term rotation to
*each* term in turn, `n_steps` times over the total duration `t` -- the first-order
product-formula approximation to `exp(-i*H*t)` for the *whole* `terms` sum, which
generally doesn't commute term-by-term. `terms` must be in dict form here (`{0: 'Z', 1:
'Z'}`, not `'ZZ'`) -- unlike [`observables`](observables.md)'s functions, this one
doesn't accept the string shorthand. Fidelity against the exact result climbs from
`0.74` at 1 step to `0.9995` at 20 -- more slices trade circuit depth for accuracy, the
same tradeoff every Trotterized-circuit algorithm makes.

## Step 3. A statevector under a pulse, no circuit at all

```python
import jax.numpy as jnp
from dense_evolution.circuits.trotter import continuous_pulse_evolve

X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
psi_final, _ = continuous_pulse_evolve(
    psi0=jnp.array([1.0, 0.0], dtype=jnp.complex128),
    hamiltonian_fn=lambda coeff: coeff * X,
    coeffs_t=jnp.ones(100),
    dt=0.01,
)
psi_final
```

```
Array([0.540306+0.j       , 0.      -0.8414727j], dtype=complex64)
```

`continuous_pulse_evolve(psi0, hamiltonian_fn, coeffs_t, dt)` evolves a statevector
directly through `jax.lax.scan`, one `exp(-i*H(coeff)*dt)` slice per entry of
`coeffs_t` -- no Python-side gate list ever built, so a finely-resolved pulse (many
slices) costs compile time, not accumulating memory the way Step 2's growing `ops` list
would. `hamiltonian_fn` maps a single coefficient to the instantaneous Hamiltonian
matrix; a constant `coeffs_t` (100 slices of `1.0`, above) is the simplest case, a
plain `X` rotation for total time `1.0` -- matching `scipy.linalg.expm(-1j*X*1.0)`
applied to `|0>` to four decimal places. A real, non-constant `coeffs_t` (a smooth pulse
envelope, a transient burst) works the same way.

## Step 4. A density matrix under a dissipative channel

```python
from dense_evolution.circuits.trotter import continuous_dissipative_evolve

rho0 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
rho_final, _ = continuous_dissipative_evolve(
    rho0=rho0,
    channel_fn=de.global_depolarizing_channel,
    params_t=jnp.ones(50) * 0.05,
)
rho_final
```

```
Array([[0.5384719 +0.j, 0.        +0.j],
       [0.        +0.j, 0.46152753+0.j]], dtype=complex64)
```

Not every real time-dependent process is coherent -- a cosmic-ray impact on a
superconducting chip, say, transiently collapses `T1` in a way no Hermitian
`hamiltonian_fn` can express. `continuous_dissipative_evolve(rho0, channel_fn,
params_t)` is `continuous_pulse_evolve`'s dissipative counterpart: `channel_fn` applies
an arbitrary CPTP map (here, [`global_depolarizing_channel`](noise.md) at a constant
`p=0.05`) once per slice directly to the density matrix. Starting from the pure state
`|0><0|`, 50 slices of depolarizing noise drag it most of the way to maximally mixed
(`I/2`) -- exactly the decay a real dissipative process produces.

---

## Details

**`observable_fn`**: both `continuous_pulse_evolve` and `continuous_dissipative_evolve`
accept an optional `observable_fn`, returned as the second element of the tuple (`None`
above, since neither call passed one) -- when given, it's evaluated at every slice and
the full history is returned alongside the final state/density matrix, for watching a
quantity evolve over the pulse instead of only reading its endpoint.

**Where the time-dependent case came from**: `continuous_pulse_evolve`/
`continuous_dissipative_evolve` were generalized out of ad hoc pulse/channel-evolution
code first written for
[Dense-Evolution-Discovery Experiment 33](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/germanium_iswap_validation/)
(a real 56ns raised-cosine baseband iSWAP pulse, arXiv:2608.16716) and reused for
[Experiment 34](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cosmic_ray_burst_validation/)'s
reproduction of a real cosmic-ray-induced error burst (arXiv:2104.05219, the `T1`-collapse
example above).

::: dense_evolution.circuits.trotter

---

**See also**: [`fermions`](fermions.md) and [`entropy`](entropy.md), the other two
modules promoted alongside this one from a real traversable-wormhole-inspired quantum
teleportation reproduction (arXiv:2604.10090). `dashboard_core.wormhole.run_wormhole_protocol_trotter`
uses this module's Trotterized circuit (Step 2) as the "closer to real hardware"
backend, cross-verified against the exact-evolution backend -- see
[Dense-Evolution-Discovery](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the real experiments (run with the exact backend, for scan speed).
