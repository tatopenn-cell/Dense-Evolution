"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.parser as part of the Phase 2 subpackage split
(see prog.txt). Kept so `from dense_evolution.parser import QASMParser`
(used by external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.parser directly in new code.
"""
from dense_evolution.circuits.parser import QASMParser, QASMCircuit

__all__ = ['QASMParser', 'QASMCircuit']
