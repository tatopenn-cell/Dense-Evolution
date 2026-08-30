"""Mitigation subpackage: Zero-Noise Extrapolation (zne) and the
predictive-healing primitives (healing) it composes, plus density-matrix
diagnostics validated in Dense-Evolution-Discovery: the sandwiched
quantum Renyi divergence (renyi), magic entropy (magic_entropy), a
classical-shadows-based estimator for magic entropy from measurement
snapshots (magic_entropy_shadows), a multi-qubit per-state magic monotone
(stabilizer_renyi_entropy -- NOT the same quantity as magic_entropy, see
that module's own docstring for the distinction), and the classical
Kullback-Leibler divergence over probability distributions (kl_divergence)."""
from .zne import (
    richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
    bounded_exponential_extrapolate,
    project_to_physical, uhlmann_fidelity, zne_density_matrix,
    jsd_predictive_zne_density_matrix,
    richardson_extrapolate_jit, zero_noise_extrapolation_jit,
    polynomial_extrapolate_jit, uhlmann_fidelity_jit, zne_density_matrix_jit,
)
from .healing import (
    calculate_advanced_sigma, calculate_phi_ab, calculate_vettore_dinamico,
    calculate_vettore_statico, calculate_delta_preemp, evaluate_phi_trigger,
    calculate_jax_reflection, MemoryReflectionEngine, GLOBAL_CONSTANTS,
)
from .renyi import sandwiched_renyi_divergence, sandwiched_renyi_divergence_jit
from .magic_entropy import magic_entropy, magic_entropy_jit
from .stabilizer_renyi_entropy import stabilizer_renyi_entropy, stabilizer_renyi_entropy_jit
from .magic_entropy_shadows import (
    sample_classical_shadow, magic_entropy_from_shadows,
    approx_shadow_std, fit_shadow_sample_complexity,
)
from .kl_divergence import kl_divergence, kl_divergence_jit

__all__ = [
    "richardson_extrapolate", "zero_noise_extrapolation", "polynomial_extrapolate",
    "bounded_exponential_extrapolate",
    "project_to_physical", "uhlmann_fidelity", "zne_density_matrix",
    "jsd_predictive_zne_density_matrix",
    "richardson_extrapolate_jit", "zero_noise_extrapolation_jit",
    "polynomial_extrapolate_jit", "uhlmann_fidelity_jit", "zne_density_matrix_jit",
    "calculate_advanced_sigma", "calculate_phi_ab", "calculate_vettore_dinamico",
    "calculate_vettore_statico", "calculate_delta_preemp", "evaluate_phi_trigger",
    "calculate_jax_reflection", "MemoryReflectionEngine", "GLOBAL_CONSTANTS",
    "sandwiched_renyi_divergence", "sandwiched_renyi_divergence_jit",
    "magic_entropy", "magic_entropy_jit",
    "stabilizer_renyi_entropy", "stabilizer_renyi_entropy_jit",
    "sample_classical_shadow", "magic_entropy_from_shadows",
    "approx_shadow_std", "fit_shadow_sample_complexity",
    "kl_divergence", "kl_divergence_jit",
]
