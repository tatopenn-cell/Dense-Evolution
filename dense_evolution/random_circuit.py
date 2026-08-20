"""Backward-compatibility shim -- the real implementation moved to
dense_evolution.circuits.random_circuit as part of fixing the module/
function name collision (dense_evolution.qft/random_circuit each had a
function sharing its own module's name -- any code importing the flat
submodule path directly clobbered the package's re-exported function
attribute with the module object, e.g. `TypeError: 'module' object is
not callable`; see prog.txt). Kept so
`from dense_evolution.random_circuit import random_circuit` (used by
external consumers, e.g. Dense-Evolution-Discovery) keeps working
unchanged. Import from dense_evolution.circuits.random_circuit directly
in new code.
"""
from dense_evolution.circuits.random_circuit import random_circuit

__all__ = ['random_circuit']
