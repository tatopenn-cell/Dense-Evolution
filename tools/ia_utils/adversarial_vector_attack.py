"""
Gradient-based adversarial stress-testing for enhanced_dense_healing_hybrid's
Phi-Trigger decision -- a targeted, crafted perturbation instead of the
random NaN/Inf corruption this healing pipeline is normally tested
against, adapted from IGME's chained-differentiable-attack idea
(arXiv:2607.27465, "IGME: Efficient Chained Method Ensemble for
Transferable Semantic Segmentation Attacks") applied to vector sequences
instead of image segmentation.

evaluate_phi_trigger (dense_evolution.healing) makes its keep-vs-median-
replace decision by thresholding |v_dinamic| against NON_STATIC_THRESHOLD_A
-- a hard step, not differentiable at the boundary. But v_dinamic itself
(via calculate_phi_ab -> calculate_vettore_dinamico) is built entirely
from norms, dot products, log, and clip -- all JAX-differentiable. This
crafts a minimal perturbation to a single vector in the sequence, via
gradient ascent/descent on |v_dinamic| (projected back into an epsilon
L2-ball each step, the standard PGD pattern), that flips the trigger's
decision at that point:

- "flip_to_dynamic": push an originally-static point (would be replaced
  by the local median) across the threshold so the trigger keeps it
  as-is instead -- the more security-relevant direction, since it
  represents a worst-case corruption crafted to *evade* the healer by
  looking like genuine motion, rather than obvious noise.
- "flip_to_static": push an originally-dynamic point (would be kept)
  across the threshold so the trigger discards it as noise instead --
  the failure mode of genuine signal getting wrongly suppressed.

This does not attack the median-filter fallback itself, only the
Phi-Trigger's keep-vs-replace decision -- see craft_adversarial_healing_
perturbation's own docstring for what "success" means precisely.
"""

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from dense_evolution.healing import calculate_phi_ab, calculate_vettore_dinamico, GLOBAL_CONSTANTS

__all__ = ['craft_adversarial_healing_perturbation']


def _trigger_magnitude(state_B, state_A, ipg_vector):
    """|v_dinamic| -- the continuous quantity evaluate_phi_trigger
    thresholds. Differentiable w.r.t. state_B (the vector being attacked);
    state_A/ipg_vector are treated as fixed context, not attacked."""
    phi_ab = calculate_phi_ab(state_A, state_B, ipg_vector)
    E_A = jnp.linalg.norm(state_A)
    E_B = jnp.linalg.norm(state_B)
    v_dinamic = calculate_vettore_dinamico(E_A, E_B, phi_ab)
    return jnp.abs(v_dinamic)


_trigger_magnitude_grad = jax.grad(_trigger_magnitude, argnums=0)


def craft_adversarial_healing_perturbation(
    vettori: np.ndarray,
    target_idx: int,
    radius_baseline: Optional[int] = None,
    epsilon: float = 0.1,
    n_steps: int = 50,
    step_size: Optional[float] = None,
    direction: str = "flip_to_dynamic",
) -> dict:
    """Crafts a minimal adversarial perturbation to vettori[target_idx],
    within an L2 epsilon-ball, that flips enhanced_dense_healing_hybrid's
    Phi-Trigger decision at that index -- a targeted stress test, not
    random noise.

    Reproduces enhanced_dense_healing_hybrid's own per-step computation
    at target_idx exactly (same baseline_mean window, same adaptive
    radius default, same inter-point-gradient vector) so the crafted
    perturbation is faithful to what the real healing pipeline would
    actually see, not a simplified stand-in.

    BUG FIX: step_size used to default to 2*epsilon/n_steps -- tying the
    per-step move size to the epsilon budget. Verified directly this
    makes LARGER epsilon give WORSE (higher final |v_dinamic|) results,
    not better: a bigger budget means a bigger step, which overshoots
    and oscillates around the minimum instead of converging to it,
    which is the opposite of what a bigger budget should ever do for a
    correct optimizer. step_size is now independent of epsilon (a small
    fixed default, tuned to v_dinamic's typical local scale) -- epsilon
    only bounds where the iterate is allowed to end up (via projection
    after each step), not how big each step is.

    Args:
        vettori: array-like, shape (n_steps, dim), the (unperturbed)
            vector sequence. Not modified in place.
        target_idx: index to attack; must be >= 2 (the healing loop's
            own starting point) and < len(vettori).
        radius_baseline: same meaning as enhanced_dense_healing_hybrid's
            own parameter; None uses the same adaptive default.
        epsilon: L2-norm budget for the perturbation (the attack is
            projected back into this ball after every gradient step).
        n_steps: number of PGD-style gradient steps.
        step_size: per-step move size along the normalized gradient;
            None uses a small fixed default (0.02) independent of
            epsilon -- see the bug-fix note above for why.
        direction: "flip_to_dynamic" (evade -- make static-looking
            input pass through unhealed) or "flip_to_static" (suppress
            -- make dynamic-looking input get median-replaced instead).

    Returns:
        dict with:
            perturbed_vettori: copy of vettori with vettori[target_idx]
                replaced by the crafted perturbation.
            success: bool, True only if the trigger decision actually
                flipped in the requested direction (a small epsilon or
                too few steps can fail to cross the threshold).
            original_trigger_active / final_trigger_active: bool, the
                Phi-Trigger's decision (True = dynamic/kept) before and
                after the perturbation.
            original_magnitude / final_magnitude: float, |v_dinamic|
                before and after.
            perturbation_norm: float, actual L2 norm of the applied
                perturbation (<= epsilon).
    """
    vettori = np.asarray(vettori, dtype=np.float64)
    n, hidden_dim = vettori.shape
    if target_idx < 2 or target_idx >= n:
        raise ValueError(f"target_idx must be in [2, {n - 1}], got {target_idx} (n={n})")
    if direction not in ("flip_to_dynamic", "flip_to_static"):
        raise ValueError(f"direction must be 'flip_to_dynamic' or 'flip_to_static', got {direction!r}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    if radius_baseline is None:
        radius_baseline = min(20, max(3, n // 3))
    lo = max(0, target_idx - radius_baseline)

    state_A = jnp.array(np.mean(vettori[lo:target_idx], axis=0))
    ipg_raw = vettori[target_idx - 1] - vettori[target_idx - 2]
    norm_ipg_raw = np.linalg.norm(ipg_raw)
    ipg_vector = jnp.array(ipg_raw / norm_ipg_raw) if norm_ipg_raw > 1e-9 else jnp.array(ipg_raw)

    original_state_B = jnp.array(vettori[target_idx])
    threshold = GLOBAL_CONSTANTS['NON_STATIC_THRESHOLD_A']
    original_magnitude = float(_trigger_magnitude(original_state_B, state_A, ipg_vector))
    original_trigger_active = original_magnitude > threshold

    sign = 1.0 if direction == "flip_to_dynamic" else -1.0
    if step_size is None:
        step_size = 0.02

    # Track the best iterate seen (most favorable to `direction`), not
    # just the last one -- with a fixed small step size the trajectory
    # can overshoot past the optimum and drift back worse on later
    # steps, especially once it's pinned against the epsilon-ball
    # boundary; keeping the best-so-far makes the result robust to
    # exactly how many steps were requested.
    state_B = original_state_B
    best_state_B = original_state_B
    best_magnitude = original_magnitude
    for _ in range(n_steps):
        grad = _trigger_magnitude_grad(state_B, state_A, ipg_vector)
        grad_norm = jnp.linalg.norm(grad)
        step = jnp.where(grad_norm > 1e-12, grad / grad_norm, jnp.zeros_like(grad))
        state_B = state_B + sign * step_size * step
        # Project back into the epsilon L2-ball around the original vector.
        delta = state_B - original_state_B
        delta_norm = jnp.linalg.norm(delta)
        state_B = jnp.where(delta_norm > epsilon, original_state_B + delta / delta_norm * epsilon, state_B)

        current_magnitude = float(_trigger_magnitude(state_B, state_A, ipg_vector))
        is_better = (current_magnitude > best_magnitude) if direction == "flip_to_dynamic" else (current_magnitude < best_magnitude)
        if is_better:
            best_magnitude = current_magnitude
            best_state_B = state_B

    state_B = best_state_B
    final_magnitude = float(_trigger_magnitude(state_B, state_A, ipg_vector))
    final_trigger_active = final_magnitude > threshold
    perturbation_norm = float(jnp.linalg.norm(state_B - original_state_B))

    perturbed_vettori = vettori.copy()
    perturbed_vettori[target_idx] = np.asarray(state_B)

    flipped = final_trigger_active != original_trigger_active
    success = flipped and (
        (direction == "flip_to_dynamic" and final_trigger_active)
        or (direction == "flip_to_static" and not final_trigger_active)
    )

    return {
        "perturbed_vettori": perturbed_vettori,
        "success": bool(success),
        "original_trigger_active": bool(original_trigger_active),
        "final_trigger_active": bool(final_trigger_active),
        "original_magnitude": original_magnitude,
        "final_magnitude": final_magnitude,
        "perturbation_norm": perturbation_norm,
    }
