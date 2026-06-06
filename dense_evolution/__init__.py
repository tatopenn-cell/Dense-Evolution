"""
Dense-Evolution
High-performance quantum statevector simulator optimized for NISQ circuits.
"""

from .simulator import DenseSVSimulator
from .parser import QASMParser, QASMCircuit
from .compiler import QuantumTranspiler
from .registry import NoiseModel, QuantumHardwareRegistry
from .gates import GATES, PARAMETRIC_GATES, GATE_IDS

<<<<<<< HEAD
__version__ = "8.1.1"
=======
__version__ = "8.1.2"
>>>>>>> 10dd0b7 (v8.1.2 - SafeMemoryGuard Anti-OOM, chunk.py rewrite, README update)
