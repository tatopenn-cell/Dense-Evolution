"""Mitigation subpackage: Zero-Noise Extrapolation (zne) and the
predictive-healing primitives (healing) it composes, plus two
density-matrix diagnostics validated in Dense-Evolution-Discovery: the
sandwiched quantum Renyi divergence (renyi) and magic entropy
(magic_entropy)."""
from .zne import (
    richardson_extrapolate, zero_noise_extrapolation, polynomial_extrapolate,
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

__all__ = [
    "richardson_extrapolate", "zero_noise_extrapolation", "polynomial_extrapolate",
    "project_to_physical", "uhlmann_fidelity", "zne_density_matrix",
    "jsd_predictive_zne_density_matrix",
    "richardson_extrapolate_jit", "zero_noise_extrapolation_jit",
    "polynomial_extrapolate_jit", "uhlmann_fidelity_jit", "zne_density_matrix_jit",
    "calculate_advanced_sigma", "calculate_phi_ab", "calculate_vettore_dinamico",
    "calculate_vettore_statico", "calculate_delta_preemp", "evaluate_phi_trigger",
    "calculate_jax_reflection", "MemoryReflectionEngine", "GLOBAL_CONSTANTS",
    "sandwiched_renyi_divergence", "sandwiched_renyi_divergence_jit",
    "magic_entropy", "magic_entropy_jit",
]
