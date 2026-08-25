# Dashboard Core — Vector Healing (Composer's healing panel)

Real predictive-healing pass over a noisy vector sequence (VQE/MD
telemetry, quantum state trajectories, or any other `(n_steps, dim)`
array) — a thin dashboard-facing wrapper around
[`ia_utils.vector_healing.enhanced_dense_healing_hybrid`](ia_utils_vector_healing.md),
lazily imported so a missing/stripped `ia_utils` install fails with a
clear error at call time rather than breaking `dashboard_core` import
for everyone.

```python
import numpy as np
from dashboard_core.vector_healing import run_vector_healing

rng = np.random.default_rng(0)
vectors = rng.normal(0, 1, size=(30, 3))
vectors[10, 1] += 8.0    # a noise spike -- healed by the per-step Phi-Trigger
vectors[15, 0] = np.nan  # genuine NaN corruption -- this is what fallback_triggered reports

result = run_vector_healing(vectors)
print(result.fallback_triggered)  # True -- because of the NaN, NOT the spike (see note below)
print(result.healed_vectors[10])  # [0.115, -0.338, -0.044] -- spike replaced by the local median
print(vectors[10].tolist())       # [-1.010, 7.791, -0.159] -- the raw, unhealed value for comparison
```

`fallback_triggered` is narrower than it sounds: it's `True` only when genuine NaN/Inf
corruption was present and corrected, not whenever the Phi-Trigger replaces an outlier
spike with the local median — a spike-only run above (no NaN/Inf) heals `vectors[10]`
exactly the same way but reports `fallback_triggered=False`.

::: dashboard_core.vector_healing

---

**Not to be confused with** [`ia_utils.vector_healing`](ia_utils_vector_healing.md)
(same name, different module) — that one has the real
`median_healing`/`enhanced_dense_healing_hybrid` implementation; this
one is the dashboard's request/response wrapper around it.
