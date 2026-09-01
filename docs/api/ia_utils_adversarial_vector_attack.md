# IA Utils — Adversarial Vector Attack

[`enhanced_dense_healing_hybrid`](ia_utils_vector_healing.md) decides, per step,
whether a change in a vector sequence looks like genuine dynamics or static noise --
this module asks how easy that decision is to fool. A gradient-based (PGD-style) stress
test crafts the *minimal* perturbation, within a fixed budget, that flips the
Phi-Trigger's decision either direction -- not random noise thrown at the problem and
hoping.

## Step 1. Disguise a static point as genuine dynamics

```python
import numpy as np
from ia_utils.adversarial_vector_attack import craft_adversarial_healing_perturbation

vettori = np.array([[1.0, 1.0]] * 10) + np.random.default_rng(0).normal(0, 0.001, (10, 2))
result = craft_adversarial_healing_perturbation(vettori, target_idx=5, direction='flip_to_dynamic')

result['success'], result['original_trigger_active'], result['final_trigger_active']
```

```
(True, False, True)
```

`vettori` is a nearly flat sequence -- real Phi-Trigger noise, not corrupted at all
(`original_trigger_active=False`, correctly seen as static). `craft_adversarial_healing_perturbation(
vettori, target_idx, direction='flip_to_dynamic')` searches for the smallest
perturbation (within an epsilon-ball, default `epsilon=0.1`) to step 5 that flips the
trigger on -- and finds one: `final_trigger_active=True`, `success=True`. This is the
*evade* direction: a real attacker's goal would be making corrupted data pass through
[`enhanced_dense_healing_hybrid`](ia_utils_vector_healing.md) unhealed by disguising it
as legitimate dynamics.

## Step 2. The other direction: suppress a genuine trigger

```python
vettori2 = np.array([[1.0, 1.0]] * 10) + np.random.default_rng(0).normal(0, 0.01, (10, 2))
result2 = craft_adversarial_healing_perturbation(vettori2, target_idx=3, direction='flip_to_static')

result2['success'], result2['original_trigger_active'], result2['final_trigger_active']
```

```
(True, True, False)
```

This time step 3 already genuinely triggers (`original_trigger_active=True` -- real
noise at this magnitude, not a hand-picked corruption). `direction='flip_to_static'`
finds the opposite perturbation: one that makes the Phi-Trigger wrongly see this real
trigger as noise (`final_trigger_active=False`) and median-replace it instead -- the
*suppress* direction, a real signal getting wiped out rather than a fake one getting
through. Not every target index is this exploitable within the epsilon budget (some
points genuinely resist the attack, `success=False`) -- which points are vulnerable
depends on the sequence's own local shape around them, not a fixed property of the
attack itself.

---

## Details

**Why this is possible at all**: `evaluate_phi_trigger` ([`healing`](healing.md))
thresholds `|v_dinamic|` with a hard step (not differentiable at the boundary), but
`v_dinamic` itself is built entirely from JAX-differentiable operations
(`calculate_phi_ab`/`calculate_vettore_dinamico`) -- so a gradient-based search can
still find the minimal input change that pushes the pre-threshold value across the
boundary, even though the boundary decision itself isn't smooth.

**Two real bugs found and fixed during this utility's own verification, not assumed
correct**: the default `step_size` used to scale with the `epsilon` budget -- a
*larger* budget converged to a *worse* result (verified directly, non-monotonic in
epsilon), now a small fixed default independent of `epsilon`. And `calculate_phi_ab`'s
`[0,1]` clip saturates for inputs whose semantic distance exceeds
`MAX_SEMANTIC_DISTANCE`, giving an exact-zero gradient -- a real property of the
formula, not a bug in the attack, but one that used to be silently misreported as "no
better point found" rather than detected and reported (`perturbation_norm == 0`,
`success == False`).

Adapted from IGME's chained-differentiable-attack idea
([arXiv:2607.27465](https://arxiv.org/abs/2607.27465), "Efficient Chained Method
Ensemble for Transferable Semantic Segmentation Attacks", He & Zhang), applied here to
vector sequences instead of image segmentation.

::: ia_utils.adversarial_vector_attack

---

**See also**: [`ia_utils.vector_healing`](ia_utils_vector_healing.md) for the function
under test, and [`dense_evolution.healing`](healing.md) for the underlying Phi-Trigger
primitives.
