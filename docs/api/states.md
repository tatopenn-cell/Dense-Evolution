# States (common state-preparation circuits)

A GHZ state -- `(|00...0> + |11...1>) / sqrt(2)`, every qubit perfectly correlated
with every other one -- is the standard multi-qubit entanglement benchmark: it shows
up at the top of practically every experiment and test script in this package,
usually hand-written as `[('h', 0), ('cx', 0, 1), ('cx', 1, 2), ...]`. `ghz_state` is
that snippet, written once.

## Step 1. Build and run a GHZ state

```python
import numpy as np
import dense_evolution as de

sim = de.DenseSVSimulator(3)
sim.run_circuit(de.ghz_state(3))
print(de.ghz_state(3))
print(np.round(sim.get_probabilities(), 4))
```

```
[('h', 0), ('cx', 0, 1), ('cx', 1, 2)]
[0.5 0.  0.  0.  0.  0.  0.  0.5]
```

`de.ghz_state(3)` is a plain gate-tuple list -- `H` on qubit 0, then a linear chain of
`cx` gates propagating that superposition outward one qubit at a time -- so it drops
straight into `run_circuit` like any hand-built circuit. Only index `0` (`|000>`) and
index `7` (`|111>`) carry probability, each exactly `0.5`: measuring always gives all
zeros or all ones, never a mix.

## Step 2. What noise does to it

```python
from dense_evolution.registry import NoiseModel

noisy_sv = NoiseModel.apply_to_sv(
    np.asarray(sim.get_statevector()), n=3, model='depolarizing', p=0.05,
    rng=np.random.default_rng(0),
)
print(np.round(np.abs(noisy_sv) ** 2, 4))
```

```
[0.   0.   0.5  0.   0.   0.5  0.   0.  ]
```

Depolarizing noise at `p=0.05` on this particular random draw flipped one qubit,
moving all the probability from `|000>`/`|111>` to `|010>`/`|101>` -- a stark,
worst-case-looking result from a single 3-qubit sample, not a general "GHZ states are
fragile" statement; see [Noise](noise.md) for the full model and how averaging over
many trajectories (as [ZNE](mitigation.md) does) recovers a smooth error curve instead
of one noisy sample like this.

---

## Details

**Requires `n_qubits >= 2`** -- a single qubit has no partner to entangle with, so an
`n=1` "GHZ state" is undefined and raises `ValueError`.

**Implementation**: `[('h', 0)] + entangling_layer(n_qubits, pattern='linear',
gate='cx')` -- `ghz_state` is a thin, named wrapper around
[`entangling_layer`](topology.md)'s `'linear'` pattern, not a separate
implementation. Building the same superposition-then-chain idea with a different
topology (e.g. `pattern='star'` for a hub-and-spoke GHZ variant) means calling
`entangling_layer` directly instead.

**See also**: [Topology](topology.md) for `entangling_layer` and its other four
connectivity patterns; [QFT](qft.md) for the other standard textbook circuit builder
in this package.

::: dense_evolution.physics.states
