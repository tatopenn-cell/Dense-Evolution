# Noise

A real quantum computer is never perfect. `dense_evolution.noise` has three
ways to put that imperfection into a simulation, depending on what you need
from it.

## Step 1. Build a circuit

```python
import numpy as np
import dense_evolution as de

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(2)
sim.run_circuit(circuit.to_tuples())
sv = np.asarray(sim.get_statevector())
```

`sv` is the exact, noiseless result. Every step below starts from it.

## Step 2. Add noise to it

```python
from dense_evolution.noise import NoiseModel

rng = np.random.default_rng(0)
sv_noisy = NoiseModel.apply_to_sv(sv.copy(), 2, "depolarizing", 0.1, rng=rng)
round(float(np.vdot(sv_noisy, sv_noisy).real), 4)
```

```
1.0
```

`NoiseModel.apply_to_sv` takes the statevector, the qubit count, which error
model to simulate, and how strong it is. The result is still a valid,
correctly normalised state — noise redistributes probability, it never
breaks it.

## Step 3. See every available model

```python
NoiseModel.MODELS
```

```
['ideal', 'depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping', 'combined']
```

Swap `"depolarizing"` above for any other name to use it instead. `p` is
that model's error probability, except for `"amplitude_damping"`, where `p`
is the decay rate.

```python
NoiseModel.kraus_description("bitflip")["physical"]
```

```
'Bit flip σ_x with probability p'
```

## Step 4. Use a real device's own noise instead of a made-up number

```python
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from dense_evolution.interop import noise_model_from_qiskit_backend

backend = FakeSherbrooke()
spec = noise_model_from_qiskit_backend(backend)
len(spec)
```

```
652
```

Each entry in `spec` is a real, measured error rate for one gate on one
qubit of that device — pass any one of them as `model`/`p`/`qubits` to
`NoiseModel.apply_to_sv` above, instead of guessing a number.

## Step 5. Make the noise strength part of a gradient

```python
import jax
from dense_evolution.noise import NoiseSpec

key = jax.random.PRNGKey(0)
spec = NoiseSpec(model="depolarizing", p=0.05, jax_key=key, qubits=[0, 1])
spec
```

```
NoiseSpec(model='depolarizing', p=0.05, qubits=(0, 1))
```

Pass a `NoiseSpec` as `circuit_to_energy_fn`'s `noise=` argument to trace
noise strength through `jax.grad` along with every other parameter, instead
of applying it as a separate step outside the gradient.

## Step 6. Search for a worst-case coherent error against a QEC code

The three steps above are all stochastic, single-qubit errors. This one is
different: a coherent error spread across every qubit at once, searched for
by gradient ascent instead of sampled at random.

```python
import jax.numpy as jnp
from dense_evolution.noise.coherent_attack import craft_adversarial_delta_constrained

x_stabilizers = ["IIIXXXX", "IXXIIXX", "XIXIXIX"]
sv_code = jnp.ones(2 ** 7, dtype=jnp.complex128) / jnp.sqrt(2 ** 7)
delta, leakage, _ = craft_adversarial_delta_constrained(
    sv_code, x_stabilizers, epsilon=0.5, linf_cap=0.1, n_steps=50
)
delta.shape
```

```
(7,)
```

`delta[q]` is the coherent rz angle the search found for qubit `q`. The
`linf_cap` argument matters: without it (`craft_adversarial_delta`, no
per-qubit cap), the search degenerately concentrates the whole error on one
qubit — which a real decoder corrects exactly every time, making that
"worst case" actually harmless. Capping each qubit's share forces the
search into directions that spread across multiple qubits, which is where
real decoder failures happen.

## Step 7. Simulate a cosmic-ray burst hitting the chip

A cosmic ray or gamma ray striking the chip briefly raises every qubit's
decay rate, then lets it fall back to normal — this profile reproduces a
real measured event from a published 26-qubit device.

```python
import numpy as np
from dense_evolution.mitigation.zne import cosmic_ray_burst_profile

baseline = 0.001
times_us = np.array([0.0, 10.0, 1000.0, 50000.0])
profile = cosmic_ray_burst_profile(times_us, baseline_gamma=baseline)
[round(float(x), 6) for x in profile]
```

```
[0.001, 0.002487, 0.003599, 0.001372]
```

At `t=0` (the impact instant) the decay rate is still just `baseline`. It
climbs to over 3x baseline within about a millisecond, then relaxes back
down over tens of milliseconds. Feed the result into
`continuous_dissipative_evolve` alongside `amplitude_damping_channel` to
inject a real burst shape into a circuit or QEC study, instead of a
constant noise rate.

## Step 8. Apply noise to a density matrix instead of a statevector

Steps 2-7 above all work on a statevector. Density-matrix ZNE (see
[Density-matrix ZNE healing](../examples.md#density-matrix-zne-healing))
needs noise applied directly to a density matrix instead.

```python
rho = np.outer(sv, sv.conj())

from dense_evolution.noise import global_depolarizing_channel

rho_noisy = global_depolarizing_channel(rho, 0.1)
round(float(np.trace(rho_noisy).real), 6)
```

```
1.0
```

`global_depolarizing_channel` mixes the *whole* register toward the fully
mixed state as one unit — a different physical process from `NoiseModel`'s
`"depolarizing"`, which acts independently per qubit. Use this one for a
state-prep/measurement (SPAM) error reported as a single number over the
whole register, not per-qubit gate noise. `amplitude_damping_channel(rho,
gamma)` is the density-matrix equivalent of Step 3's `"amplitude_damping"`,
single-qubit only, and is what Step 7's burst profile is meant to drive.

## Step 9. Make the noise strength oscillate instead of scaling smoothly

```python
from dense_evolution.noise import oscillating_p_eff

[round(float(oscillating_p_eff(base_p=0.1, factor=f, freq=2.0, amp=0.5)), 4) for f in [0.0, 1.0, 2.0, 3.0]]
```

```
[0.1, 0.15, 0.1, 0.05]
```

Most mitigation techniques (Richardson extrapolation, for one) assume noise
grows smoothly as you scale it up. `oscillating_p_eff` builds a
deliberately non-smooth noise-vs-scale relationship instead, to check
whether a technique still works when that assumption doesn't hold.

---

## Details

### Kraus formulas

| Model | Kraus operators |
| :--- | :--- |
| `depolarizing` | `{√(1-p)I, √(p/3)X, √(p/3)Y, √(p/3)Z}` |
| `bitflip` | `{√(1-p)I, √p·X}` |
| `phaseflip` | `{√(1-p)I, √p·Z}` |
| `amplitude_damping` | `K0=diag(1,√(1-γ)), K1=[[0,√γ],[0,0]]` |
| `combined` | `depolarizing(p/2)` then `amplitude_damping(p/3)` |
| `ideal` | identity |

Every channel draws one fire/no-fire decision per qubit per shot (plus one
Pauli choice for `depolarizing`/`combined`'s depolarizing sub-step) —
the same single-Pauli-per-qubit-per-shot convention STIM's `DEPOLARIZE1(p)`
uses. Prior to v8.1.57, each channel instead drew one independent decision
per computational-basis amplitude pair, which over-decohered entangled
states (a measured value dropped from 1.0 to 0.31 at p=0.15 on one test
case) — fixed by drawing one decision per qubit and applying it uniformly.

### Photon loss is not a separate channel

Photon loss on a dual-rail-encoded qubit is exactly this library's
`amplitude_damping` channel (Step 3 above) — there is no separate photon
noise model, and none is needed.

### Moved here from mitigation.zne

`global_depolarizing_channel`, `amplitude_damping_channel`,
`cosmic_ray_burst_profile`, and `oscillating_p_eff` used to live in
`dense_evolution.mitigation.zne` — they generate noise, they don't
mitigate it, so that was the wrong home. `dense_evolution.mitigation.zne`
still re-exports all four for backward compatibility.

### Coherent adversarial noise: an honest negative result

`craft_adversarial_delta` (no L-infinity cap) was tested against a real
decoder for the Steane [[7,1,3]] code and found to converge to a direction
with **zero** real decoder-failure rate — a coherent error concentrated on
a single qubit always collapses to something a distance-3 code corrects
exactly, so the search's own "worst case" is actually safe. Random noise
directions of the same L2 budget, spread across multiple qubits, failed the
decoder readily by comparison. `craft_adversarial_delta_constrained`'s
L-infinity cap fixes this by forbidding that degenerate solution. Promoted
from Dense-Evolution-Discovery's Steane investigation, generalized from a
fixed 7-qubit table to any stabilizer list.

::: dense_evolution.noise

## See Also

- [`DenseSVSimulator`](simulator.md), [`QASMParser`](parser.md) — build the
  circuit and statevector every step above starts from.
- [`dense_evolution.mitigation`](mitigation.md) — correcting noise after
  it's applied, instead of just simulating it.
- [`dense_evolution.qec`](qec.md) — the decoder side of the code
  `craft_adversarial_delta_constrained` attacks.
