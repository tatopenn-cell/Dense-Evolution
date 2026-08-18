"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.mitigation.healing as part of the Phase 2 subpackage
split (see prog.txt). Kept so `from dense_evolution.healing import
calculate_phi_ab` (used by external consumers, e.g. tools/ia_utils and
Dense-Evolution-Discovery) keeps working unchanged. Import from
dense_evolution.mitigation.healing directly in new code.
"""
from dense_evolution.mitigation.healing import (
    calculate_advanced_sigma, calculate_phi_ab, calculate_vettore_dinamico,
    calculate_vettore_statico, calculate_delta_preemp, evaluate_phi_trigger,
    calculate_jax_reflection, MemoryReflectionEngine, GLOBAL_CONSTANTS,
)

__all__ = [
    "calculate_advanced_sigma", "calculate_phi_ab", "calculate_vettore_dinamico",
    "calculate_vettore_statico", "calculate_delta_preemp", "evaluate_phi_trigger",
    "calculate_jax_reflection", "MemoryReflectionEngine", "GLOBAL_CONSTANTS",
]
