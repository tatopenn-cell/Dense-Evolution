# Healing (predictive primitives)

> The shared decision primitive [Mitigation](mitigation.md) and
> [Vector Healing](ia_utils_vector_healing.md) both call into — see
> [Concepts](../concepts.md) for which of those two you actually want.

Predictive-healing primitives for noisy vector sequences (VQE/MD
telemetry, quantum state trajectories): a "Phi-Trigger" that, given a
state and a local baseline, decides whether an observed change looks
like genuine dynamics or static noise, plus supporting sync/reflection
functions. Built empirically, one formula at a time — see the module's
own docstring for which parts carry a real information-theoretic reading
(`calculate_vettore_dinamico`'s core term is a log-likelihood ratio) and
which are a geometric construction instead (`calculate_phi_ab`), so as
not to overclaim either way.

**Applied layer**: [`ia_utils.vector_healing.enhanced_dense_healing_hybrid`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/tools/ia_utils/vector_healing.py)
is what actually calls these primitives on a real `(n_steps, dim)`
sequence — per step, runs the Phi-Trigger against a local baseline
window and either keeps the value (genuine dynamics) or replaces it with
the local median (static noise), after sanitizing any NaN/Inf first
regardless of that decision. This shipped with the pre-rebuild
dashboard_core's Streamlit "AI healing shield" middleware (VQE/MD
telemetry routed through it before any panel was built from it), and
was left behind — not removed — when dashboard_core was rebuilt around
the Composer kernel.

**Reintegrated end to end**: `dashboard_core.run_vector_healing` (a thin
wrapper, mirroring `mitigation.py`'s shape) → the kernel's
`POST /api/vector_healing` → the MCP tool `dense_evolution_vector_healing`
(see `mcp_server/README.md`). All three call the same real primitives on
this page — no separate reimplementation.

---

**See also**: [`dense_evolution.mitigation`](mitigation.md) — `zero_noise_extrapolation`'s
healing-adapted branch (triggered by passing `sigma_at_base_noise`) calls
`calculate_delta_preemp` from this module. Unlike `run_vector_healing`
above, this branch is **not** currently reachable from the kernel or MCP:
`dashboard_core.run_zne_mitigation` never passes `sigma_at_base_noise`,
and the kernel's `MitigateRequest` has no field for it. This was
originally left as a known follow-up pending `calculate_advanced_sigma`'s
undefined input provenance — that question has since been closed, not
completed: Dense-Evolution-Discovery Experiment 35
(`scripts/zne_healing_sigma_provenance.py`) fed the branch a real,
oracle-free `sigma_at_base_noise` (the empirical std of the noisy trial
ensemble) and found, via a permutation-test negative control, that the
branch's coefficient perturbation doesn't discriminate real sigma from
randomly shuffled sigma at all — a confound, not a usable signal. Wiring
this branch up would not have helped even with fully-designed inputs.
`calculate_advanced_sigma` is now deprecated (`DeprecationWarning`,
kept for backward compatibility only) rather than completed.
