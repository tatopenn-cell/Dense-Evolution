"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.registry as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.registry import HAS_JAX`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.registry directly in new code.
"""
from dense_evolution.circuits.registry import HAS_JAX, NoiseModel, NoiseSpec, QuantumHardwareRegistry

__all__ = ['HAS_JAX', 'NoiseModel', 'NoiseSpec', 'QuantumHardwareRegistry']
