# Healing (predictive primitives)

> The shared decision primitive [Mitigation](mitigation.md) and
> [Vector Healing](ia_utils_vector_healing.md) both call into -- see
> [Concepts](../concepts.md) for which of those two you actually want.

Given one step of a noisy telemetry sequence (a VQE energy, an MD trajectory value), is
this a genuine change worth keeping, or a spike worth smoothing away? This module's
primitives answer that question one number at a time -- the "Phi-Trigger" -- and are
what [`ia_utils.vector_healing.enhanced_dense_healing_hybrid`](ia_utils_vector_healing.md)
calls internally on a whole sequence. Reach for this page directly when you want that
same decision on raw values of your own, without going through the full sequence-healing
wrapper.

## Step 1. The trigger: real change or noise?

```python
import jax.numpy as jnp
import dense_evolution.healing as h

dq_dt = jnp.array([0.001, 0.002, 0.5, 0.001])
h.evaluate_phi_trigger(dq_dt)
```

```
(Array([0., 0., 1., 0.], dtype=float32, weak_type=True),
 Array([0.15, 0.15, 0.05, 0.15], dtype=float32, weak_type=True),
 Array([0.11, 0.11, 0.01, 0.11], dtype=float32, weak_type=True))
```

`dq_dt` is one rate-of-change value per step -- how much a quantity moved since the
previous step. `evaluate_phi_trigger` returns three arrays: the trigger itself (`1.0`
where a step's rate crosses the "this is real dynamics" threshold, `0.0` otherwise --
only the third step here, `0.5`, is large enough), and two damping coefficients that
drop when the trigger fires (`0.15` -> `0.05` and `0.11` -> `0.01`) -- a genuine change
gets less aggressive smoothing applied around it than a static step would.

## Step 2. Where `dq_dt` comes from: comparing two real states

```python
ipg_vector = jnp.array([1.0, 0.0])
phi_ab = h.calculate_phi_ab(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]), ipg_vector)

v_stable = h.calculate_vettore_dinamico(jnp.array(1.0), jnp.array(1.001), phi_ab)
v_jump = h.calculate_vettore_dinamico(jnp.array(1.0), jnp.array(1.5), phi_ab)

h.evaluate_phi_trigger(jnp.array([v_stable, v_jump]))[0]
```

```
Array([0., 1.], dtype=float32, weak_type=True)
```

Step 1's `dq_dt` isn't usually handed to you directly -- it's built from two states
`E_A`/`E_B` (a scalar energy or observable at consecutive steps) plus `Phi_AB`, an
alignment/coherence factor between them (`calculate_phi_ab`, here computed once for two
identical direction vectors and reused for both comparisons).
`calculate_vettore_dinamico(E_A, E_B, Phi_AB)` is `log(E_B/E_A)` scaled by that
alignment -- a log-likelihood-ratio-flavored measure of how much `E_A` moved to become
`E_B`. `1.0 -> 1.001` (`v_stable = 0.0035`) doesn't trigger; `1.0 -> 1.5` (`v_jump =
1.42`) does -- exactly the same 0/1 split Step 1 showed directly, now built from two
real states instead of a rate-of-change handed in already computed.

---

## Details

**What's principled vs. empirical here**: `calculate_vettore_dinamico`'s core term is a
genuine log-likelihood ratio (the same elementary quantity Kullback-Leibler divergence
is built from -- see [`kl_divergence`](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/kullback_leibler_divergence/)
for the distinction between this one un-weighted scalar ratio and a full KL divergence
over a probability distribution). `calculate_phi_ab` is a geometric construction instead,
built empirically rather than derived from an information-theoretic quantity -- worth
knowing before leaning on either reading too heavily.

**Applied layer**: [`ia_utils.vector_healing.enhanced_dense_healing_hybrid`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/tools/ia_utils/vector_healing.py)
is what actually calls these primitives on a real `(n_steps, dim)` sequence -- see
[Vector Healing](ia_utils_vector_healing.md) for that page's own worked examples. This
shipped with the pre-rebuild dashboard_core's Streamlit "AI healing shield" middleware
(VQE/MD telemetry routed through it before any panel was built from it), and was left
behind -- not removed -- when dashboard_core was rebuilt around the Composer kernel.

**Reintegrated end to end**: `dashboard_core.run_vector_healing` (a thin wrapper,
mirroring `mitigation.py`'s shape) -> the kernel's `POST /api/vector_healing` -> the
MCP tool `dense_evolution_vector_healing` (see `mcp_server/README.md`). All three call
the same real primitives on this page -- no separate reimplementation.

**The other branch, not wired up**: [`dense_evolution.mitigation`](mitigation.md)'s
`zero_noise_extrapolation` has a healing-adapted branch (triggered by passing
`sigma_at_base_noise`) that calls `calculate_delta_preemp` from this module. Unlike
`run_vector_healing` above, this branch is **not** currently reachable from the kernel
or MCP: `dashboard_core.run_zne_mitigation` never passes `sigma_at_base_noise`, and the
kernel's `MitigateRequest` has no field for it. This was originally left as a known
follow-up pending `calculate_advanced_sigma`'s undefined input provenance -- that
question has since been closed, not completed: Dense-Evolution-Discovery Experiment 35
(`scripts/zne_healing_sigma_provenance.py`) fed the branch a real, oracle-free
`sigma_at_base_noise` (the empirical std of the noisy trial ensemble) and found, via a
permutation-test negative control, that the branch's coefficient perturbation doesn't
discriminate real sigma from randomly shuffled sigma at all -- a confound, not a usable
signal. Wiring this branch up would not have helped even with fully-designed inputs.
`calculate_advanced_sigma` is now deprecated (`DeprecationWarning`, kept for backward
compatibility only) rather than completed -- excluded from the guide above for that
reason.

::: dense_evolution.healing
