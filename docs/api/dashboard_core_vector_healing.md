# Dashboard Core — Vector Healing (Composer's healing panel)

Real predictive-healing pass over a noisy vector sequence (VQE/MD
telemetry, quantum state trajectories, or any other `(n_steps, dim)`
array) — a thin dashboard-facing wrapper around
[`ia_utils.vector_healing.enhanced_dense_healing_hybrid`](ia_utils_vector_healing.md),
lazily imported so a missing/stripped `ia_utils` install fails with a
clear error at call time rather than breaking `dashboard_core` import
for everyone.

::: dashboard_core.vector_healing

---

**Not to be confused with** [`ia_utils.vector_healing`](ia_utils_vector_healing.md)
(same name, different module) — that one has the real
`median_healing`/`enhanced_dense_healing_hybrid` implementation; this
one is the dashboard's request/response wrapper around it.
