# IA Utils — Vector Sequence Healing

> Correcting a *numeric log/trajectory*, not a quantum measurement result — see
> [Concepts](../concepts.md) if you're looking for [Mitigation](mitigation.md) instead.

Many pipelines produce one vector per step: an energy at every VQE iteration, a
position at every MD step, an embedding per token. If even one of those steps is
corrupted — a numerical overflow, a `NaN`/`Inf` recovered upstream, a solver that
silently failed at one point — plotting or analyzing the sequence as-is drags the
whole scale off with it. `ia_utils.vector_healing` looks at each step in turn and
decides whether the change from the previous one is real dynamics (left alone) or
an isolated glitch (replaced with the local median of the nearby steps).

Healing is the inverse of noise: noise corrupts a value, healing tries to undo
that. [`dense_evolution.mitigation`](mitigation.md) (Zero-Noise Extrapolation) is
the same idea applied to a *quantum measurement result* — an expectation value
distorted by real hardware/simulated noise. This module applies it to a
*sequence of vectors* instead — a log, a trajectory, an embedding stream. Same
inverse-of-noise idea, two different objects; one is not a substitute for the
other.

## Step 1. A small sequence with one value out of place

```python
import numpy as np
from ia_utils.vector_healing import enhanced_dense_healing_hybrid

v = np.array([[1.0, 2.0], [1.1, 2.1], [1.05, 2.05], [50.0, -30.0], [1.08, 2.08], [1.02, 2.02]])
clean, telem = enhanced_dense_healing_hybrid(v)
clean.round(3)
```

```
array([[1.  , 2.  ],
       [1.1 , 2.1 ],
       [1.05, 2.05],
       [1.05, 2.05],
       [1.1 , 2.05],
       [1.08, 2.05]])
```

Six steps, each a 2-value vector drifting slowly upward — except row 3, which
jumps to `[50, -30]` and back down again on row 4. `enhanced_dense_healing_hybrid`
replaces only that one row with a value close to its neighbors; every other row
is untouched, including the genuine upward drift the sequence is actually doing.
`telem` (returned alongside `clean`) reports what happened —
`telem["reconstruction_error"]` is the total distance between the input and the
healed output, non-zero here because something really was corrected. The same
call works on any sequence shaped this way — a VQE energy log with one corrupted
line reshaped to `(n_iterations, 1)`, for instance, heals exactly the same way.

## Step 2. A dissociation curve with one geometry that didn't converge

A real, chemistry-specific case: scanning a bond-length dissociation curve one
geometry at a time, where a real solver can genuinely fail to converge at a
particular geometry (near-degenerate orbitals, a bad initial guess for that one
point) and return a value far off the otherwise smooth potential-energy curve —
a well-known headache in quantum chemistry, not a hypothetical.

```python
import numpy as np
from dashboard_core.hamiltonians import ground_state_energy_sparse
from ia_utils.vector_healing import enhanced_dense_healing_hybrid

rs = np.linspace(0.5, 2.5, 11)
scan = np.array([ground_state_energy_sparse(["H", "H"], [[0, 0, 0], [0, 0, r]]) for r in rs])
scan = scan.reshape(-1, 1)
scan[5, 0] = 0.0
clean, telem = enhanced_dense_healing_hybrid(scan)
clean.ravel().round(5)
```

```
array([-1.05516, -1.13619, -1.12056, -1.07919, -1.03519, -1.07919, -0.97143,
       -0.95434, -0.94437, -0.93892, -0.93605])
```

`ground_state_energy_sparse` (see [Hamiltonians](dashboard_core_hamiltonians.md))
gives the real H2 ground-state energy at each of 11 bond lengths from 0.5 to 2.5
Å — a real dissociation curve, smooth after its minimum near the true equilibrium
bond length. Point 5 (r=1.5 Å) is overwritten with `0.0`, standing in for that one
geometry's solver failing to converge. Healing puts it back near the curve
(`-1.07919`, the local median of its neighbors — close to, though not identical
to, the true `-0.99815` that solver would have returned if it had converged: a
median is an estimate from nearby points, not a re-run of the failed calculation)
while leaving the other 10 real points on the curve untouched.

## See Also

- [`dense_evolution.mitigation`](mitigation.md) — the inverse-of-noise idea
  applied to a quantum measurement result instead of a vector sequence.
- [`dense_evolution.healing`](healing.md) — the predictive "Phi-Trigger"
  primitives `enhanced_dense_healing_hybrid` calls internally to decide, per
  step, whether an observed change looks like genuine dynamics or static noise.
- [Hamiltonians](dashboard_core_hamiltonians.md) — `ground_state_energy_sparse`,
  the source of Step 2's real dissociation curve.
- [`ia_utils.adversarial_vector_attack`](ia_utils_adversarial_vector_attack.md) —
  a gradient-based robustness test of that same Phi-Trigger decision.

---

## Details

### `median_healing` vs. `enhanced_dense_healing_hybrid`

`median_healing` always applies a median filter to every step, with no
notion of "genuine vs. corrupted" — a plain, unconditional smoothing pass.
`enhanced_dense_healing_hybrid` is the one worth reaching for in practice:
it only replaces a step when its own decision rule (`trigger_mode`, below)
judges that step to be noise, leaving every other step bit-for-bit as it was.
Both preprocess `NaN`/`Inf` first (column-mean imputation) so a corrupted
value never propagates into a healthy neighbor's own repair.

### `trigger_mode`: `'phi'` vs. `'adaptive'`

`'phi'` (the default) is the original Phi-Trigger
(`dense_evolution.mitigation.healing.evaluate_phi_trigger`): a fixed threshold,
`|v_dinamic| > 0.01`, on the normalized step-to-step change. `'adaptive'` adjusts
that threshold to the sequence's own local variability instead of a fixed
constant. Both call the same underlying decision machinery and, on every real
sequence tried while writing this page, gave identical output — the difference
only shows up on sequences whose natural noise level is far from what the fixed
0.01 threshold assumes.

### When this does nothing, on purpose

Fed a sequence of TF-IDF vectors from consecutive chunks of a real paper (this
project's own local `quantumrag` index, 76 chunks from Pednault et al. 2019 —
see [Chunk](chunk.md)'s disk-overflow section for why that specific paper
mattered here), `enhanced_dense_healing_hybrid` left every single chunk
unchanged, in both `trigger_mode`s. That's the correct outcome, not a bug: two
consecutive chunks of running text about different subsections of a paper are
*supposed* to look very different in vector space — that's a real topic change,
not corruption, and nothing here should try to smooth it away. The same applies
to Step 2's own dissociation curve before the corrupted point was added to it —
zero points changed on the clean curve. Reaching for this module makes sense
once there's an actual bad reading in the sequence, not merely a bumpy but
genuine one.

### `reconstruction_error` and `fallback_triggered`

`telem["reconstruction_error"]` is `0.0` whenever nothing was changed (both
cases in the previous section) and positive whenever at least one step was
replaced — a quick way to check, in code, whether healing actually did
anything to a given sequence without diffing the arrays by hand.
`telem["fallback_triggered"]` records whether the hybrid strategy fell back to
a plain median pass instead of the Phi-Trigger decision — see
`enhanced_dense_healing_hybrid`'s own docstring below for exactly when that
happens.

::: ia_utils.vector_healing
