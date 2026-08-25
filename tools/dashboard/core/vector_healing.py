"""
Real predictive-healing pass over a noisy vector sequence (VQE/MD
telemetry, quantum state trajectories, or any other (n_steps, dim) array)
-- a thin wrapper around `ia_utils.vector_healing.enhanced_dense_healing_hybrid`,
which itself is built on `dense_evolution.healing`'s Phi-Trigger primitives
(calculate_phi_ab, calculate_vettore_dinamico, evaluate_phi_trigger).

This existed in the pre-rebuild dashboard_core (Streamlit dashboard's
"AI healing shield" middleware, routing VQE/MD telemetry through it
before any panel was built from it) but was left behind when
dashboard_core was rebuilt around the Composer kernel -- see this
package's __init__.py docstring ("will be reintegrated selectively once
this base is solid"). Reintegrated here as its own module, mirroring
mitigation.py's shape (a dataclass result + one thin run_* function),
rather than resurrecting the old monolithic dashboard_core.py.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ia_utils.vector_healing import enhanced_dense_healing_hybrid
except ImportError as _import_error:
    # ia_utils ships in the same distribution as dashboard_core (see
    # pyproject.toml's packages list) so this shouldn't fail on a normal
    # install -- but ia_utils.vector_healing's own dependency chain
    # (dense_evolution.healing, which needs JAX) can still fail
    # transitively, or a stripped/vendored copy could omit ia_utils
    # entirely. Deferred to call time instead of failing this whole
    # module's import, which would otherwise take down every unrelated
    # dashboard_core feature at import time for a healing-specific gap.
    enhanced_dense_healing_hybrid = None
    _IMPORT_ERROR = _import_error
else:
    _IMPORT_ERROR = None

__all__ = ['VectorHealingResult', 'run_vector_healing']


@dataclass
class VectorHealingResult:
    healed_vectors: list          # (n_steps, dim), same shape as input
    fallback_triggered: bool      # True only if genuine NaN/Inf corruption was present AND corrected
    adaptive_radius_used: int
    reconstruction_error: float   # mean per-step norm of (healed - sanitized) vectors


def run_vector_healing(vectors: np.ndarray, radius_baseline: Optional[int] = None) -> VectorHealingResult:
    """Heal a noisy (n_steps, dim) vector sequence: per step, a Phi-Trigger
    (dense_evolution.healing) decides whether the change from a local
    baseline looks like genuine dynamics (kept as-is) or static noise
    (replaced by the local median) -- see
    ia_utils.vector_healing.enhanced_dense_healing_hybrid's own docstring
    for the full algorithm. NaN/Inf entries are sanitized first
    (column-mean imputation) regardless of the trigger's decision.

    Args:
        vectors: array-like, shape (n_steps, dim), n_steps >= 0.
        radius_baseline: fixed radius for the local baseline window; if
            None (default), computed adaptively as
            min(20, max(3, n_steps // 3)).

    Returns:
        VectorHealingResult

    Examples
    --------
    >>> import numpy as np
    >>> from dashboard_core.vector_healing import run_vector_healing
    >>> rng = np.random.default_rng(0)
    >>> vectors = rng.normal(0, 1, size=(30, 3))
    >>> vectors[10, 1] += 8.0    # a noise spike -- the Phi-Trigger heals this
    >>> vectors[15, 0] = np.nan  # genuine NaN corruption -- this is what fallback_triggered reports
    >>> result = run_vector_healing(vectors)
    >>> result.fallback_triggered  # True only because of the NaN, not the spike
    True
    >>> abs(result.healed_vectors[10][1]) < 1.0  # spike replaced by the local median regardless
    True
    """
    if enhanced_dense_healing_hybrid is None:
        raise ImportError(
            "run_vector_healing requires ia_utils.vector_healing, which failed "
            f"to import ({_IMPORT_ERROR}). It ships with dense-evolution's own "
            "packages, so this usually means a stripped/vendored install is "
            "missing it, or one of its own dependencies (dense_evolution.healing "
            "needs JAX) isn't available."
        )

    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2D (n_steps, dim), got shape {arr.shape}")

    healed, metadata = enhanced_dense_healing_hybrid(arr, radius_baseline=radius_baseline)

    return VectorHealingResult(
        healed_vectors=healed.tolist(),
        fallback_triggered=bool(metadata['fallback_triggered']),
        adaptive_radius_used=int(metadata['adaptive_radius_used']),
        reconstruction_error=float(metadata['reconstruction_error']),
    )
