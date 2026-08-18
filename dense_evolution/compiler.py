"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.compiler as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.compiler import QuantumTranspiler`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.compiler directly in new code.
"""
from dense_evolution.circuits.compiler import QuantumTranspiler

__all__ = ['QuantumTranspiler']
