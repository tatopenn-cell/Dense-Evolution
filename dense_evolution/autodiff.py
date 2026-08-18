"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.solvers.autodiff as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.autodiff import
circuit_to_energy_fn` (used by external consumers, e.g.
Dense-Evolution-Discovery) keeps working unchanged. Import from
dense_evolution.solvers.autodiff directly in new code.
"""
from dense_evolution.solvers.autodiff import circuit_to_energy_fn

__all__ = ["circuit_to_energy_fn"]
