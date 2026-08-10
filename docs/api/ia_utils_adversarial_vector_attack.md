# IA Utils — Adversarial Vector Attack

A gradient-based (PGD-style) stress test for
[`enhanced_dense_healing_hybrid`](ia_utils_vector_healing.md)'s
Phi-Trigger decision. `evaluate_phi_trigger`
([`dense_evolution.healing`](healing.md)) thresholds `|v_dinamic|` with a
hard step (not differentiable at the boundary), but `v_dinamic` itself is
built entirely from JAX-differentiable operations — this crafts the
*minimal* perturbation (projected into an L2 epsilon-ball) that flips the
trigger either direction, rather than adding random noise and hoping.

Adapted from IGME's chained-differentiable-attack idea
([arXiv:2607.27465](https://arxiv.org/abs/2607.27465), "Efficient Chained
Method Ensemble for Transferable Semantic Segmentation Attacks", He &
Zhang), applied here to vector sequences instead of image segmentation.

::: ia_utils.adversarial_vector_attack

---

**Two directions of attack**:

- `flip_to_dynamic` (evade) — make static-looking corruption pass through
  unhealed.
- `flip_to_static` (suppress) — make genuine dynamic signal get wrongly
  median-replaced.

Two real bugs were found and fixed during this utility's own
verification, not assumed correct: the default `step_size` used to scale
with the epsilon budget (a *larger* budget converged to a *worse*
result — verified directly, non-monotonic in epsilon — now a small fixed
default independent of epsilon), and `calculate_phi_ab`'s `[0,1]` clip
saturating for inputs whose semantic distance exceeds
`MAX_SEMANTIC_DISTANCE`, giving an exact-zero gradient (a real property
of the formula, now detected and reported — `perturbation_norm == 0`,
`success == False` — rather than silently misreported as "no better
point found").

**See also**: [`ia_utils.vector_healing`](ia_utils_vector_healing.md) for
the function under test, and [`dense_evolution.healing`](healing.md) for
the underlying Phi-Trigger primitives.
