# Dashboard Core — Mitigation (Composer's ZNE panel)

Real Zero-Noise Extrapolation (ZNE) error mitigation, wired to the same
engine and real noise channels ([`dense_evolution.registry.NoiseModel`](registry.md))
the rest of Composer uses — a thin dashboard-facing layer over
[`dense_evolution.mitigation`](mitigation.md), not a separate
reimplementation.

::: dashboard_core.mitigation

---

**Not to be confused with** [`dense_evolution.mitigation`](mitigation.md)
(same name, different module) — that one is the actual ZNE
implementation (`richardson_extrapolate`, `zne_density_matrix`,
`jsd_predictive_zne_density_matrix`, ...); this one is the dashboard's
request/response wrapper around it.
